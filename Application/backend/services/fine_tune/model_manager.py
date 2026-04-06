"""
CaneNexus – Model Manager
Handles model versioning, in-place replacement, hot-swap, and rollback.

Convention:
    - Active model is always at the standard path: steel.pth / sugar.pth
    - Archived versions: steel_old_001.pth, steel_old_002.pth, etc.
    - The original base model becomes *_old_001.pth on first fine-tune.
"""

import shutil
import re
from pathlib import Path
from datetime import datetime

import torch

from config import (
    STEEL_MODEL_PATH, SUGAR_MODEL_PATH,
    STEEL_MODEL_DIR, SUGAR_MODEL_DIR,
)
from database.mongo_client import model_versions_collection
from database.schemas import create_model_version_document


def _get_model_dir(domain: str) -> Path:
    """Return the model directory for the given domain."""
    if domain == "steel":
        return STEEL_MODEL_DIR
    elif domain == "sugar":
        return SUGAR_MODEL_DIR
    raise ValueError(f"Unknown domain: {domain}")


def _get_model_path(domain: str) -> Path:
    """Return the active model file path for the given domain."""
    if domain == "steel":
        return STEEL_MODEL_PATH
    elif domain == "sugar":
        return SUGAR_MODEL_PATH
    raise ValueError(f"Unknown domain: {domain}")


def _get_next_archive_version(domain: str) -> int:
    """Determine the next archive version number by scanning existing files."""
    model_dir = _get_model_dir(domain)
    prefix = "steel_old_" if domain == "steel" else "sugar_old_"
    pattern = re.compile(rf"^{prefix}(\d{{3}})\.pth$")

    max_version = 0
    for f in model_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            v = int(match.group(1))
            if v > max_version:
                max_version = v

    return max_version + 1


def get_active_version(domain: str) -> int:
    """Get the currently active model version number from MongoDB."""
    doc = model_versions_collection.find_one(
        {"domain": domain, "is_active": True},
        sort=[("version", -1)]
    )
    if doc:
        return doc["version"]
    return 0  # Base model (original, never fine-tuned)


def archive_current_model(domain: str) -> str:
    """
    Archive the current active model by renaming it to *_old_NNN.pth.

    Returns:
        The archive filename (e.g., "steel_old_001.pth").
    """
    model_path = _get_model_path(domain)
    model_dir = _get_model_dir(domain)
    version = _get_next_archive_version(domain)

    prefix = "steel_old_" if domain == "steel" else "sugar_old_"
    archive_filename = f"{prefix}{version:03d}.pth"
    archive_path = model_dir / archive_filename

    # Copy (not move) to preserve the active model during the swap
    shutil.copy2(str(model_path), str(archive_path))

    print(f"[CaneNexus] Archived {model_path.name} → {archive_filename}")
    return archive_filename


def save_finetuned_model(
    domain: str,
    state_dict: dict,
    finetune_job_id: str,
    metrics: dict,
    corrections_used: int
) -> int:
    """
    Save fine-tuned model as the new active model.

    Steps:
        1. Archive current model → *_old_NNN.pth
        2. Save new state_dict as steel.pth / sugar.pth
        3. Register in model_versions collection
        4. Return the new version number

    Args:
        domain: "steel" or "sugar"
        state_dict: The full model state dict to save
        finetune_job_id: Job ID that produced this model
        metrics: Training metrics dict
        corrections_used: Number of corrections used in training

    Returns:
        New version number.
    """
    model_path = _get_model_path(domain)

    # 1. Archive current
    parent_version = get_active_version(domain)
    archive_filename = archive_current_model(domain)

    # If this is the first fine-tune, register the base model as version 1
    if parent_version == 0:
        base_doc = create_model_version_document(
            version=1,
            domain=domain,
            checkpoint_filename=archive_filename,
            parent_version=0,
            finetune_job_id="base_model",
            metrics={},
            corrections_used=0,
            is_active=False
        )
        try:
            model_versions_collection.insert_one(base_doc)
        except Exception as e:
            print(f"[CaneNexus] Base version registration failed: {e}")
        parent_version = 1
    else:
        # parent_version > 0: The previous active version was just archived.
        # We MUST update its checkpoint_filename in the database.
        try:
            model_versions_collection.update_one(
                {"domain": domain, "version": parent_version},
                {"$set": {"checkpoint_filename": archive_filename}}
            )
        except Exception as e:
            print(f"[CaneNexus] Failed to update parent version filename: {e}")

    # 2. Save new model
    torch.save(state_dict, str(model_path))
    print(f"[CaneNexus] Saved fine-tuned model → {model_path.name}")

    # 3. Register new version
    new_version = parent_version + 1

    # Deactivate all previous versions for this domain
    model_versions_collection.update_many(
        {"domain": domain},
        {"$set": {"is_active": False}}
    )

    version_doc = create_model_version_document(
        version=new_version,
        domain=domain,
        checkpoint_filename=f"{domain}.pth",
        parent_version=parent_version,
        finetune_job_id=finetune_job_id,
        metrics=metrics,
        corrections_used=corrections_used,
        is_active=True
    )

    try:
        model_versions_collection.insert_one(version_doc)
    except Exception as e:
        print(f"[CaneNexus] Version registration failed: {e}")

    return new_version


def rollback_model(domain: str, target_version: int) -> dict:
    """
    Rollback the active model to a specified previous version.

    Steps:
        1. Archive current active model
        2. Copy the target version checkpoint → steel.pth / sugar.pth
        3. Update version registry
        4. Hot-swap weights in memory

    Args:
        domain: "steel" or "sugar"
        target_version: Version number to rollback to

    Returns:
        Dict with status and version info.
    """
    model_path = _get_model_path(domain)
    model_dir = _get_model_dir(domain)

    # Find the target version document
    target_doc = model_versions_collection.find_one({
        "domain": domain,
        "version": target_version
    })

    if not target_doc:
        raise ValueError(f"Version {target_version} not found for domain {domain}")

    target_checkpoint = target_doc["checkpoint_filename"]
    target_path = model_dir / target_checkpoint

    if not target_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {target_checkpoint}"
        )

    # 1. Archive current
    archive_filename = archive_current_model(domain)

    # Register the archived version
    current_version = get_active_version(domain)
    if current_version > 0:
        # The current active is being archived — its version stays in the registry
        # We MUST update its checkpoint_filename in the database to the archive file.
        model_versions_collection.update_one(
            {"domain": domain, "version": current_version},
            {"$set": {"checkpoint_filename": archive_filename}}
        )

    # 2. Copy target → active
    shutil.copy2(str(target_path), str(model_path))
    print(f"[CaneNexus] Restored {target_checkpoint} → {model_path.name}")

    # 3. Update version registry
    model_versions_collection.update_many(
        {"domain": domain},
        {"$set": {"is_active": False}}
    )
    model_versions_collection.update_one(
        {"domain": domain, "version": target_version},
        {"$set": {"is_active": True}}
    )

    # 4. Hot-swap in memory
    hot_swap_from_disk(domain)

    return {
        "status": "rolled_back",
        "domain": domain,
        "restored_version": target_version,
        "archived_as": archive_filename
    }


def hot_swap_from_disk(domain: str):
    """
    Reload the active model weights from disk into the in-memory model.
    Called after save or rollback.
    """
    from models.loader import get_model, get_device

    model = get_model()
    device = get_device()
    model_path = _get_model_path(domain)

    state_dict = torch.load(str(model_path), map_location=device)

    if domain == "sugar":
        # Load just the components that may have been fine-tuned
        if "sugar_head.weight" in state_dict:
            # This is a head-only checkpoint
            model.sugar_head.load_state_dict({
                k.replace("sugar_head.", ""): v
                for k, v in state_dict.items()
                if k.startswith("sugar_head.")
            })
        else:
            # Full model state dict — reload everything
            model.load_state_dict(state_dict, strict=False)

    elif domain == "steel":
        if "seg_head.weight" in state_dict:
            model.seg_head.load_state_dict({
                k.replace("seg_head.", ""): v
                for k, v in state_dict.items()
                if k.startswith("seg_head.")
            })
        else:
            model.load_state_dict(state_dict, strict=False)

    model.eval()
    print(f"[CaneNexus] Hot-swapped {domain} model from disk")


def list_versions(domain: str = None) -> list:
    """List all model versions, optionally filtered by domain."""
    query = {}
    if domain:
        query["domain"] = domain

    docs = list(
        model_versions_collection.find(query)
        .sort("version", -1)
    )

    for doc in docs:
        doc["_id"] = str(doc["_id"])
        if doc.get("created_at"):
            doc["created_at"] = doc["created_at"].isoformat()

    return docs
