"""
Steel Defect Detection Inference
    - Runs the DDA-ViT model on steel strip image
    - Returns structured results

Refer to Code/DDA-ViT/DDA-ViT.py --> predict_steel()
"""

import os
import time
import cv2
import numpy as np

from config import STEEL_CLASSES
from models.loader import get_model, get_device
from inference.image_utils import sliding_window_inference, generate_mask_overlay


def predict_steel(image_path: str) -> dict:
    """
    Runs steel defect segmentation on a single image

    Args:
        - image_path: Absolute path to the steel strip image

    Returns:
        - Structured result dict with 
            - defect summary
            - mask paths
            - time
    """
    model = get_model()
    device = get_device()

    start_time = time.time()

    # Load Image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img.copy()

    # Normalise to float32 [0, 1]
    img = img.astype(np.float32) / 255.0

    # Ensure height = 256
    img = cv2.resize(img, (img.shape[1], 256))

    # Sliding Window Inference
    probs = sliding_window_inference(model, img, device)

    # Argmax to get class mask
    mask = np.argmax(probs, axis=0)

    # Resize mask back to original dimensions
    mask = cv2.resize(
        mask.astype(np.float32),
        (original.shape[1], original.shape[0]),
        interpolation=cv2.INTER_NEAREST
    ).astype(np.int32)

    # Defect Summary
    unique, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size

    defect_summary = {}
    for i, cls_name in enumerate(STEEL_CLASSES):
        cls_idx = i + 1  # classes are 1-indexed
        area_pct = 0.0
        detected = False

        if cls_idx in unique:
            idx = list(unique).index(cls_idx)
            area_pct = round((counts[idx] / total_pixels) * 100, 4)
            detected = True

        defect_summary[f"class_{cls_name}"] = {
            "detected": detected,
            "area_pct": area_pct
        }

    # Dominant defect (class with highest area)
    defect_classes_only = {
        k: v for k, v in defect_summary.items() if v["detected"]
    }
    dominant_defect = "none"
    if defect_classes_only:
        dominant_defect = max(
            defect_classes_only, key=lambda k: defect_classes_only[k]["area_pct"]
        )

    total_defect_area = sum(
        v["area_pct"] for v in defect_summary.values() if v["detected"]
    )

    # Overlay Visualisation
    filename_prefix = os.path.splitext(os.path.basename(image_path))[0]
    overlay_filename, raw_mask_filename = generate_mask_overlay(
        original, mask, filename_prefix
    )

    # Time
    inference_time_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "image_path": image_path,
        "image_filename": os.path.basename(image_path),
        "domain": "steel",
        "defect_summary": defect_summary,
        "dominant_defect": dominant_defect,
        "total_defect_area_pct": round(total_defect_area, 4),
        "mask_overlay_path": overlay_filename,
        "raw_mask_path": raw_mask_filename,
        "inference_time_ms": inference_time_ms
    }