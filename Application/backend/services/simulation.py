"""
CaneNexus – Virtual Simulation Service
Batch processing of image directories with SSE event streaming.
"""

import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path

from config import ALLOWED_EXTENSIONS, DEFAULT_SIMULATION_LIMIT
from services.pipeline import run_pipeline
from database.mongo_client import simulations_collection
from database.schemas import create_simulation_document


def _scan_images(directory: str, limit: int = None) -> list:
    """
    Scan a directory for image files.
    For sugar, also recurse into subdirectories (class folders).

    Args:
        directory: Path to image directory
        limit: Max number of images to return (None = all)

    Returns:
        Sorted list of absolute image file paths.
    """
    images = []
    
    if not directory:
        return images
        
    dir_path = Path(directory)

    if not dir_path.exists():
        return images

    for item in sorted(dir_path.rglob("*")):
        if item.is_file() and item.suffix.lower() in ALLOWED_EXTENSIONS:
            images.append(str(item.resolve()))

    if limit and limit > 0:
        images = images[:limit]

    return images


def run_simulation_stream(
    steel_dir: str,
    sugar_dir: str,
    limit_per_domain: int = DEFAULT_SIMULATION_LIMIT
):
    """
    Generator that yields SSE-formatted events for each pipeline step.

    Each yield is a string in SSE format: "data: {json}\n\n"

    Args:
        steel_dir: Path to steel test images directory
        sugar_dir: Path to sugar test images directory
        limit_per_domain: Max images per domain (0 = all)
    """
    session_id = str(uuid.uuid4())

    # - Scan directories -
    limit = limit_per_domain if limit_per_domain > 0 else None
    steel_images = _scan_images(steel_dir, limit)
    sugar_images = _scan_images(sugar_dir, limit)

    total_steel = len(steel_images)
    total_sugar = len(sugar_images)
    total = total_steel + total_sugar

    # - Create simulation document -
    sim_doc = create_simulation_document(
        session_id=session_id,
        steel_directory=steel_dir,
        sugar_directory=sugar_dir,
        total_steel_images=total_steel,
        total_sugar_images=total_sugar
    )

    try:
        simulations_collection.insert_one(sim_doc)
    except Exception as e:
        print(f"[CaneNexus] Simulation doc insert failed: {e}")

    # - Yield simulation start event -
    yield _sse_event({
        "step": "simulation_start",
        "session_id": session_id,
        "total_steel": total_steel,
        "total_sugar": total_sugar,
        "total": total,
        "limit_per_domain": limit_per_domain
    })

    # - Build processing queue -
    queue = []
    for path in steel_images:
        queue.append((path, "steel"))
    for path in sugar_images:
        queue.append((path, "sugar"))

    # - Counters for summary -
    summary = {
        "steel": {"accept": 0, "downgrade": 0, "reject": 0, "manual_inspection": 0},
        "sugar": {"unsaturated": 0, "metastable": 0, "intermediate": 0, "labile": 0}
    }
    processed = 0
    total_inference_time = 0

    # - Process each image -
    for i, (image_path, domain) in enumerate(queue):
        filename = os.path.basename(image_path)
        index = i + 1

        # Step: Start processing this image
        yield _sse_event({
            "step": "image_start",
            "index": index,
            "total": total,
            "image": filename,
            "image_path": image_path,
            "domain": domain
        })

        try:
            # Step: Run inference
            yield _sse_event({
                "step": "inference_start",
                "index": index,
                "image": filename,
                "domain": domain
            })

            # Run the full pipeline optionally overriding gemini execution
            result = run_pipeline(image_path, domain, session_id, skip_gemini=True)

            total_inference_time += result.get("total_processing_ms", 0)

            # Step: Inference complete
            yield _sse_event({
                "step": "inference_complete",
                "index": index,
                "image": filename,
                "domain": domain,
                "prediction": result["prediction"],
                "time_ms": result["step_times"].get("inference_ms", 0)
            })

            # Step: KG complete
            yield _sse_event({
                "step": "kg_complete",
                "index": index,
                "image": filename,
                "domain": domain,
                "kg_result": result["knowledge_graph"],
                "time_ms": result["step_times"].get("kg_ms", 0)
            })

            # Step: Gemini complete
            yield _sse_event({
                "step": "gemini_complete",
                "index": index,
                "image": filename,
                "domain": domain,
                "response": result["gemini_response"][:200],  # Truncate for SSE
                "time_ms": result["step_times"].get("gemini_ms", 0)
            })

            # Step: Logged
            yield _sse_event({
                "step": "logged",
                "index": index,
                "image": filename,
                "domain": domain,
                "log_id": result["log_id"]
            })

            # Update summary counters
            _update_summary(summary, domain, result)
            processed += 1

            # Calculate ETA
            avg_time = total_inference_time / processed if processed > 0 else 0
            remaining = total - processed
            eta_ms = avg_time * remaining

            # Step: Image complete
            yield _sse_event({
                "step": "image_complete",
                "index": index,
                "total": total,
                "image": filename,
                "domain": domain,
                "log_id": result["log_id"],
                "total_ms": result["total_processing_ms"],
                "progress": round(processed / total, 4),
                "processed": processed,
                "eta_ms": round(eta_ms, 0),
                "summary": summary
            })

        except Exception as e:
            yield _sse_event({
                "step": "image_error",
                "index": index,
                "image": filename,
                "domain": domain,
                "error": str(e)[:200]
            })
            processed += 1

    # - Update simulation document -
    try:
        simulations_collection.update_one(
            {"session_id": session_id},
            {"$set": {
                "status": "completed",
                "completed_at": datetime.utcnow(),
                "processed_count": processed,
                "summary": summary
            }}
        )
    except Exception as e:
        print(f"[CaneNexus] Simulation update failed: {e}")

    # - Final event -
    yield _sse_event({
        "step": "simulation_complete",
        "session_id": session_id,
        "total_processed": processed,
        "total_time_ms": round(total_inference_time, 2),
        "summary": summary
    })


def _update_summary(summary: dict, domain: str, result: dict):
    """Update the running summary counters based on pipeline result."""
    if domain == "steel":
        kg = result.get("knowledge_graph", {})
        decision = kg.get("decision", "").lower()
        if "accept" in decision:
            summary["steel"]["accept"] += 1
        elif "downgrade" in decision:
            summary["steel"]["downgrade"] += 1
        elif "reject" in decision:
            summary["steel"]["reject"] += 1
        if kg.get("requires_manual_inspection", False):
            summary["steel"]["manual_inspection"] += 1

    elif domain == "sugar":
        pred = result.get("prediction", {})
        cls = pred.get("predicted_class", "").lower()
        if cls in summary["sugar"]:
            summary["sugar"][cls] += 1


def _sse_event(data: dict) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(data, default=str)}\n\n"
