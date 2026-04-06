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
