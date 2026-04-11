"""
Sugar Quality Inspection Inference
    - Runs DDA-ViT on sugar crystallisation image
    - Returns structured results

Refer to Code/DDA-ViT/DDA-ViT.py --> predict_sugar()
"""

import os
import time
import numpy as np
import torch

from config import SUGAR_CLASSES, SUGAR_IMAGE_SIZE
from models.loader import get_model, get_device
from inference.image_utils import load_image


def predict_sugar(image_path: str) -> dict:
    """
    Run sugar crystallisation quality classification on a single image

    Args:
        - image_path: Absolute path to the sugar crystallisation image

    Returns:
        - Structured result dict with 
            - predicted class
            - confidence + probabilities
            - time
    """
    model = get_model()
    device = get_device()

    start_time = time.time()

    # Load & Preprocess
    _, img_tensor, _ = load_image(image_path, size=SUGAR_IMAGE_SIZE)
    img_tensor = img_tensor.to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(x_sugar=img_tensor)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    predicted_idx = int(np.argmax(probs))
    predicted_class = SUGAR_CLASSES[predicted_idx]
    confidence = float(probs[predicted_idx])

    # Build probabilities dict
    all_probabilities = {}
    for i, cls_name in enumerate(SUGAR_CLASSES):
        all_probabilities[cls_name] = round(float(probs[i]), 6)

    # Time
    inference_time_ms = round((time.time() - start_time) * 1000, 2)

    return {
        "image_path": image_path,
        "image_filename": os.path.basename(image_path),
        "domain": "sugar",
        "predicted_class": predicted_class,
        "confidence": round(confidence, 6),
        "all_probabilities": all_probabilities,
        "inference_time_ms": inference_time_ms
    }