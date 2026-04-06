"""
CaneNexus – Fine-Tune Job Scheduler
Manages the fine-tuning job queue using a background thread.
Only one job can run at a time.
"""

import uuid
import threading
from datetime import datetime

from database.mongo_client import (
    feedback_collection, finetune_jobs_collection
)
from database.schemas import create_finetune_job_document
from config import FINETUNE_DEFAULTS, FINETUNE_LIMITS
from models.loader import get_model, get_device
from services.fine_tune.sugar_finetune import finetune_sugar
from services.fine_tune.steel_finetune import finetune_steel
from services.fine_tune.model_manager import save_finetuned_model, hot_swap_from_disk


# - Module-level state -
_current_job = None
_job_lock = threading.Lock()


def get_current_job() -> dict | None:
    """Return the current/most-recent job status."""
    global _current_job
    if _current_job is None:
        # Load the most recent job from the DB if memory was wiped (e.g., server restart)
        recent = list(finetune_jobs_collection.find().sort("created_at", -1).limit(1))
        if recent:
            doc = recent[0]
            status = doc.get("status")
            # If a running job was interrupted by a crash, mark it correctly in memory
            if status == "running":
                status = "failed"
                doc["error_message"] = "Job interrupted by server restart."

            _current_job = {
                "job_id": doc.get("job_id"),
                "domain": doc.get("domain", ""),
                "status": status,
                "corrections_count": doc.get("corrections_count", 0),
                "config": doc.get("config", {}),
                "metrics": doc.get("metrics"),
                "model_version_created": doc.get("model_version_created"),
                "error_message": doc.get("error_message")
            }
            
            if doc.get("started_at") and hasattr(doc["started_at"], "isoformat"):
                _current_job["started_at"] = doc["started_at"].isoformat()
            if doc.get("completed_at") and hasattr(doc["completed_at"], "isoformat"):
                _current_job["completed_at"] = doc["completed_at"].isoformat()

    return _current_job


def _validate_config(config: dict) -> dict:
    """Merge user config with defaults, enforcing safety limits."""
    merged = dict(FINETUNE_DEFAULTS)
    merged.update(config)

    # Clamp values
    merged["lr"] = max(
        FINETUNE_LIMITS["lr_min"],
        min(merged["lr"], FINETUNE_LIMITS["lr_max"])
    )
    merged["epochs"] = max(
        FINETUNE_LIMITS["epochs_min"],
        min(merged["epochs"], FINETUNE_LIMITS["epochs_max"])
    )
    merged["min_corrections"] = max(
        FINETUNE_LIMITS["min_corrections_floor"],
        merged["min_corrections"]
    )

    return merged


def start_finetune_job(domain: str, config: dict = None) -> dict:
    """
    Start a fine-tuning job in a background thread.

    Args:
        domain: "sugar" or "steel"
        config: Optional hyperparameter overrides.

    Returns:
        {"job_id": str, "status": str, "message": str}
    """
    global _current_job

    if domain not in ("sugar", "steel"):
        return {"error": f"Invalid domain: {domain}", "status": "failed"}

    # Check if a job is already running
    with _job_lock:
        if _current_job and _current_job.get("status") == "running":
            return {
                "error": "A fine-tune job is already running",
                "status": "busy",
                "job_id": _current_job.get("job_id")
            }

    # Validate and merge config
    merged_config = _validate_config(config or {})
    min_corrections = merged_config["min_corrections"]

    # Fetch pending corrections
    pending = list(feedback_collection.find({
        "domain": domain,
        "status": "pending"
    }))

    if len(pending) < min_corrections:
        return {
            "error": f"Need at least {min_corrections} corrections, "
                     f"only {len(pending)} pending.",
            "status": "insufficient_data",
            "pending_count": len(pending),
            "min_required": min_corrections
        }

    # Create job
    job_id = str(uuid.uuid4())
    correction_ids = [str(doc["_id"]) for doc in pending]

    job_doc = create_finetune_job_document(
        job_id=job_id,
        domain=domain,
        config=merged_config,
        corrections_count=len(pending),
        correction_ids=correction_ids
    )

    try:
        finetune_jobs_collection.insert_one(job_doc)
    except Exception as e:
        return {"error": f"Failed to create job: {e}", "status": "failed"}

    # Set current job
    _current_job = {
        "job_id": job_id,
        "domain": domain,
        "status": "running",
        "corrections_count": len(pending),
        "config": merged_config,
        "started_at": datetime.utcnow().isoformat(),
        "progress": {"epoch": 0, "total_epochs": merged_config["epochs"]},
        "metrics": {}
    }

    # Mark corrections as being used
    from bson import ObjectId
    feedback_collection.update_many(
        {"_id": {"$in": [ObjectId(cid) for cid in correction_ids]}},
        {"$set": {"status": "used", "used_in_job_id": job_id}}
    )

    # Update job status in DB
    finetune_jobs_collection.update_one(
        {"job_id": job_id},
        {"$set": {"status": "running", "started_at": datetime.utcnow()}}
    )

    # Start background thread
    thread = threading.Thread(
        target=_run_finetune_job,
        args=(job_id, domain, pending, merged_config),
        daemon=True
    )
    thread.start()

    return {
        "job_id": job_id,
        "status": "running",
        "message": f"Fine-tune job started for {domain} with {len(pending)} corrections"
    }


def _run_finetune_job(job_id: str, domain: str, corrections: list, config: dict):
    """Background thread function that executes the fine-tuning."""
    global _current_job

    try:
        model = get_model()
        device = get_device()

        # Serialise corrections to plain dicts (remove ObjectId)
        clean_corrections = []
        for corr in corrections:
            c = dict(corr)
            c["_id"] = str(c["_id"])
            clean_corrections.append(c)

        def progress_callback(epoch, total_epochs, metrics):
            global _current_job
            if _current_job and _current_job["job_id"] == job_id:
                _current_job["progress"] = {
                    "epoch": epoch,
                    "total_epochs": total_epochs
                }
                _current_job["metrics"] = metrics

        # Run fine-tuning
        if domain == "sugar":
            result = finetune_sugar(
                model, clean_corrections, config, device, progress_callback
            )
        else:
            result = finetune_steel(
                model, clean_corrections, config, device, progress_callback
            )

        # Save the fine-tuned model
        metrics = {
            "train_loss": result["train_loss"],
            "val_loss": result["val_loss"],
            "val_accuracy": result["val_accuracy"],
            "epochs_run": result["epochs_run"]
        }

        new_version = save_finetuned_model(
            domain=domain,
            state_dict=result["state_dict"],
            finetune_job_id=job_id,
            metrics=metrics,
            corrections_used=len(corrections)
        )

        # Hot-swap the model in memory
        hot_swap_from_disk(domain)

        # Update job status
        finetune_jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "metrics": metrics,
                "model_version_created": new_version
            }}
        )

        _current_job = {
            "job_id": job_id,
            "domain": domain,
            "status": "completed",
            "corrections_count": len(corrections),
            "config": config,
            "completed_at": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "model_version_created": new_version
        }

        print(f"[CaneNexus] Fine-tune job {job_id} completed. "
              f"Model version: {new_version}")

    except Exception as e:
        error_msg = str(e)[:500]
        print(f"[CaneNexus] Fine-tune job {job_id} failed: {error_msg}")

        finetune_jobs_collection.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "failed",
                "completed_at": datetime.utcnow(),
                "error_message": error_msg
            }}
        )

        # Revert correction statuses back to pending
        from bson import ObjectId
        correction_ids = [c.get("_id") if isinstance(c.get("_id"), ObjectId)
                          else ObjectId(str(c["_id"])) for c in corrections]
        feedback_collection.update_many(
            {"_id": {"$in": correction_ids}},
            {"$set": {"status": "pending", "used_in_job_id": None}}
        )

        _current_job = {
            "job_id": job_id,
            "domain": domain,
            "status": "failed",
            "error_message": error_msg,
            "completed_at": datetime.utcnow().isoformat()
        }


def get_job_history() -> list:
    """Return all past fine-tune jobs, most recent first."""
    docs = list(
        finetune_jobs_collection.find()
        .sort("created_at", -1)
        .limit(50)
    )
    for doc in docs:
        doc["_id"] = str(doc["_id"])
        for field in ("started_at", "completed_at", "created_at"):
            if doc.get(field) and hasattr(doc[field], "isoformat"):
                doc[field] = doc[field].isoformat()
    return docs
