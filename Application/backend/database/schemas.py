"""
CaneNexus – MongoDB Document Schema Definitions
Reference structures for the documents stored in each collection.
These are not enforced by MongoDB but serve as documentation and
validation reference for the application layer.
"""

from datetime import datetime


def create_log_document(
    session_id: str,
    image_path: str,
    image_filename: str,
    domain: str,
    model_prediction: dict,
    knowledge_graph_output: dict,
    gemini_initial_response: str,
    processing_time_ms: float,
    device: str
) -> dict:
    """Create a structured log document for MongoDB insertion."""
    return {
        "session_id": session_id,
        "timestamp": datetime.utcnow(),
        "image_path": image_path,
        "image_filename": image_filename,
        "domain": domain,
        "model_prediction": model_prediction,
        "knowledge_graph_output": knowledge_graph_output,
        "gemini_initial_response": gemini_initial_response,
        "processing_time_ms": processing_time_ms,
        "metadata": {
            "model_name": "DDA-ViT",
            "steel_backbone": "mit_b4",
            "sugar_backbone": "swin_tiny_patch4_window7_224",
            "device": device
        }
    }


def create_simulation_document(
    session_id: str,
    steel_directory: str,
    sugar_directory: str,
    total_steel_images: int,
    total_sugar_images: int
) -> dict:
    """Create a structured simulation session document."""
    return {
        "session_id": session_id,
        "started_at": datetime.utcnow(),
        "completed_at": None,
        "status": "running",
        "steel_directory": steel_directory,
        "sugar_directory": sugar_directory,
        "total_steel_images": total_steel_images,
        "total_sugar_images": total_sugar_images,
        "processed_count": 0,
        "summary": {
            "steel": {
                "accept": 0,
                "downgrade": 0,
                "reject": 0,
                "manual_inspection": 0
            },
            "sugar": {
                "unsaturated": 0,
                "metastable": 0,
                "intermediate": 0,
                "labile": 0
            }
        }
    }


def create_chat_document(
    log_id: str,
    session_id: str,
    initial_message: str
) -> dict:
    """Create a chat conversation document linked to an analysis log."""
    return {
        "log_id": log_id,
        "session_id": session_id,
        "messages": [
            {
                "role": "model",
                "content": initial_message,
                "timestamp": datetime.utcnow()
            }
        ]
    }


# ── Fine-Tuning Schemas ──

def create_feedback_document(
    log_id: str,
    image_path: str,
    image_filename: str,
    domain: str,
    original_prediction: dict,
    corrected_label: dict,
    reason: str = "",
    source: str = "single_analysis"
) -> dict:
    """
    Create a feedback/correction document.

    Args:
        log_id: References the original analysis log entry.
        image_path: Absolute path to the image file.
        image_filename: Filename only.
        domain: "sugar" or "steel".
        original_prediction: Full prediction dict from the model.
        corrected_label: Domain-specific correction.
            Sugar: {"class": "metastable"}
            Steel: {"type": "region_override", "corrections": [...], "missed_defects": [...]}
        reason: Optional operator comment.
        source: "single_analysis" or "simulation_review".
    """
    return {
        "log_id": log_id,
        "image_path": image_path,
        "image_filename": image_filename,
        "domain": domain,
        "original_prediction": original_prediction,
        "corrected_label": corrected_label,
        "reason": reason,
        "source": source,
        "status": "pending",        # pending | used | discarded
        "submitted_at": datetime.utcnow(),
        "used_in_job_id": None      # set when consumed by a fine-tune job
    }


def create_finetune_job_document(
    job_id: str,
    domain: str,
    config: dict,
    corrections_count: int,
    correction_ids: list
) -> dict:
    """Create a fine-tune job tracking document."""
    return {
        "job_id": job_id,
        "domain": domain,
        "status": "queued",         # queued | running | completed | failed
        "config": config,
        "corrections_count": corrections_count,
        "correction_ids": correction_ids,
        "started_at": None,
        "completed_at": None,
        "metrics": {},              # {train_loss, val_loss, val_accuracy, epochs_run}
        "model_version_created": None,
        "error_message": None,
        "created_at": datetime.utcnow()
    }


def create_model_version_document(
    version: int,
    domain: str,
    checkpoint_filename: str,
    parent_version: int,
    finetune_job_id: str,
    metrics: dict,
    corrections_used: int,
    is_active: bool = False
) -> dict:
    """
    Create a model version registry document.

    The active model is always at the standard path (steel.pth / sugar.pth).
    Archived versions are named steel_old_NNN.pth / sugar_old_NNN.pth.
    """
    return {
        "version": version,
        "domain": domain,
        "checkpoint_filename": checkpoint_filename,
        "parent_version": parent_version,
        "finetune_job_id": finetune_job_id,
        "metrics": metrics,
        "corrections_used": corrections_used,
        "is_active": is_active,
        "created_at": datetime.utcnow()
    }

