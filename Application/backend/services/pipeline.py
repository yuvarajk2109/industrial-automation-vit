"""
Pipeline Orchestrator
    - End-to-end pipeline
        - Inference
        - Knowledge Graph
        - Gemini
        - MongoDB Log
"""

import time
import uuid

from inference.steel_inference import predict_steel
from inference.sugar_inference import predict_sugar
from knowledge_graph.steel_kg import evaluate_steel_kg
from knowledge_graph.sugar_kg import evaluate_sugar_kg
from chatbot.gemini_client import get_initial_response
from database.mongo_client import logs_collection, chats_collection
from database.schemas import create_log_document, create_chat_document
from models.loader import get_device


def _generate_programmatic_summary(domain: str, prediction: dict, kg_result: dict) -> str:
    """
    - Generates markdown summary programmatically from inference and KG results
    """
    md = "Virtual Simulation Result (Auto-Generated)\n\n"
    
    if domain == "steel":
        md += "DDA-ViT Inference (Steel)\n"
        md += f"**Defect Area Pct**: {kg_result.get('total_defect_area_pct', 0.0)}%\n"
        
        detected_classes = []
        for cls, info in prediction.get("defect_summary", {}).items():
            if info["detected"]:
                detected_classes.append(cls.replace("_", " ").title())
                
        if detected_classes:
            md += f"**Detected Classes**: {', '.join(detected_classes)}\n"
        else:
            md += "**Detected Classes**: None\n"
            
        md += "\nLogical KG Outcome\n"
        md += f"**Interpretation**: {kg_result.get('defect_interpretation', '').replace('_', ' ')}\n"
        md += f"**Quality**: {kg_result.get('quality_assessment', '').replace('_', ' ')}\n"
        md += f"**Decision**: {kg_result.get('decision', '').replace('_', ' ')}\n"
        
        if kg_result.get("requires_manual_inspection"):
            md += "\nWARNING: Manual inspection is strictly required!\n"

    elif domain == "sugar":
        md += "DDA-ViT Inference (Sugar)\n"
        md += f"**Predicted Class**: {prediction.get('predicted_class', '').title()}\n"
        md += f"**Confidence**: {prediction.get('confidence', 0.0) * 100:.2f}%\n"
        
        md += "\nLogical KG Outcome\n"
        md += f"**Supersaturation Range**: {kg_result.get('supersaturation_range', 'Unknown')}\n"
        md += f"**Nucleation Risk**: {str(kg_result.get('nucleation_risk', '')).title()}\n"
        md += f"**Growth Stability**: {str(kg_result.get('growth_stability', '')).title()}\n"
        
        actions = kg_result.get("recommended_actions", [])
        if actions:
            action_str = ', '.join(a.replace('_', ' ').title() for a in actions)
            md += f"**Recommended Actions**: {action_str}\n"
            
    return md


def run_pipeline(image_path: str, domain: str, session_id: str = None, skip_gemini: bool = False) -> dict:
    """
    - Runs full analysis pipeline for a single image

    - Steps:
        1. Runs DDA-ViT inference
        2. Evaluates Knowledge Graph
        3. Generates LLM initial response
        4. Logs to MongoDB
        5. Returns complete result

    - Args:
        - image_path: Absolute path to the image file
        - domain: "steel" or "sugar"
        - session_id: Optional session ID (for simulation grouping)
        - skip_gemini: Whether to skip querying the LLM

    - Returns:
        - Complete pipeline result dict
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    pipeline_start = time.time()
    step_times = {}

    # Inference
    t0 = time.time()
    if domain == "steel":
        prediction = predict_steel(image_path)
    elif domain == "sugar":
        prediction = predict_sugar(image_path)
    else:
        raise ValueError(f"Unknown domain: {domain}. Must be 'steel' or 'sugar'.")
    step_times["inference_ms"] = round((time.time() - t0) * 1000, 2)

    # Knowledge Graph
    t0 = time.time()
    if domain == "steel":
        kg_result = evaluate_steel_kg(prediction["defect_summary"])
    else:
        kg_result = evaluate_sugar_kg(prediction)
    step_times["kg_ms"] = round((time.time() - t0) * 1000, 2)

    # Expert Response Module
    t0 = time.time()
    if skip_gemini:
        gemini_response = _generate_programmatic_summary(domain, prediction, kg_result)
    else:
        gemini_response = get_initial_response(prediction, kg_result)
    step_times["gemini_ms"] = round((time.time() - t0) * 1000, 2)

    # Log to MongoDB
    t0 = time.time()
    device_name = str(get_device())

    log_doc = create_log_document(
        session_id=session_id,
        image_path=image_path,
        image_filename=prediction.get("image_filename", ""),
        domain=domain,
        model_prediction=prediction,
        knowledge_graph_output=kg_result,
        gemini_initial_response=gemini_response,
        processing_time_ms=round((time.time() - pipeline_start) * 1000, 2),
        device=device_name
    )

    try:
        insert_result = logs_collection.insert_one(log_doc)
        log_id = str(insert_result.inserted_id)
    except Exception as e:
        print(f"[CaneNexus] MongoDB log insert failed: {e}")
        log_id = "log_failed"

    # Create chat document
    if not skip_gemini and log_id != "log_failed":
        try:
            chat_doc = create_chat_document(
                log_id=log_id,
                session_id=session_id,
                initial_message=gemini_response
            )
            chats_collection.insert_one(chat_doc)
        except Exception as e:
            print(f"[CaneNexus] MongoDB chat insert failed: {e}")

    step_times["db_ms"] = round((time.time() - t0) * 1000, 2)

    # Total time
    total_ms = round((time.time() - pipeline_start) * 1000, 2)

    return {
        "log_id": log_id,
        "session_id": session_id,
        "domain": domain,
        "prediction": prediction,
        "knowledge_graph": kg_result,
        "gemini_response": gemini_response,
        "step_times": step_times,
        "total_processing_ms": total_ms
    }