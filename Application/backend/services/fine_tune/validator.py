"""
CaneNexus – Holdout Validator
Validates fine-tuned models against a held-out portion of corrections
to prevent overfitting and ensure quality improvement.
"""

import numpy as np
import torch
import torch.nn.functional as F

from inference.image_utils import load_image
from inference.steel_inference import predict_steel
from config import SUGAR_CLASSES, SUGAR_IMAGE_SIZE


def split_corrections(corrections: list, val_split: float = 0.2, min_val: int = 1) -> tuple:
    """
    Split corrections into training and validation sets.

    Args:
        corrections: List of feedback documents.
        val_split: Fraction to hold out for validation.
        min_val: Minimum number of validation samples.

    Returns:
        (train_corrections, val_corrections)
    """
    n = len(corrections)
    n_val = max(min_val, int(n * val_split))
    n_val = min(n_val, n - 1)  # ensure at least 1 training sample

    # Shuffle deterministically
    indices = list(range(n))
    np.random.seed(42)
    np.random.shuffle(indices)

    val_indices = set(indices[:n_val])
    train = [corrections[i] for i in range(n) if i not in val_indices]
    val = [corrections[i] for i in range(n) if i in val_indices]

    return train, val


def validate_sugar(model, val_corrections: list, device) -> dict:
    """
    Validate the fine-tuned sugar model on held-out corrections.

    Args:
        model: The DDA-ViT model instance.
        val_corrections: List of validation feedback documents.
        device: Torch device.

    Returns:
        {"val_accuracy": float, "val_loss": float, "correct": int, "total": int}
    """
    if not val_corrections:
        return {"val_accuracy": 1.0, "val_loss": 0.0, "correct": 0, "total": 0}

    model.eval()
    correct = 0
    total_loss = 0.0
    total = 0

    class_to_idx = {cls: i for i, cls in enumerate(SUGAR_CLASSES)}

    with torch.no_grad():
        for corr in val_corrections:
            image_path = corr.get("image_path", "")
            target_class = corr.get("corrected_label", {}).get("class", "")

            if not image_path or target_class not in class_to_idx:
                continue

            try:
                _, img_tensor, _ = load_image(image_path, size=SUGAR_IMAGE_SIZE)
                img_tensor = img_tensor.to(device)

                output = model(x_sugar=img_tensor)
                probs = F.softmax(output, dim=1)

                target_idx = class_to_idx[target_class]
                target_tensor = torch.tensor([target_idx], device=device)

                loss = F.cross_entropy(output, target_tensor)
                total_loss += loss.item()

                pred_idx = torch.argmax(probs, dim=1).item()
                if pred_idx == target_idx:
                    correct += 1

                total += 1
            except Exception as e:
                print(f"[CaneNexus] Validation error for {image_path}: {e}")
                continue

    if total == 0:
        return {"val_accuracy": 0.0, "val_loss": 0.0, "correct": 0, "total": 0}

    return {
        "val_accuracy": round(correct / total, 4),
        "val_loss": round(total_loss / total, 6),
        "correct": correct,
        "total": total
    }


def validate_steel(model, val_corrections: list, device) -> dict:
    """
    Validate the fine-tuned steel model on held-out corrections.
    Since steel corrections are region-level overrides (not pixel-perfect masks),
    we validate by checking if the model's prediction for the corrected regions
    now matches the corrected class.

    Returns:
        {"val_accuracy": float, "val_loss": float, "correct": int, "total": int}
    """
    if not val_corrections:
        return {"val_accuracy": 1.0, "val_loss": 0.0, "correct": 0, "total": 0}

    # For steel, validation is approximate since corrections are region-level
    # We check if the dominant defect class changed as expected
    correct = 0
    total = 0

    for corr in val_corrections:
        image_path = corr.get("image_path", "")
        corrections = corr.get("corrected_label", {}).get("corrections", [])

        if not image_path or not corrections:
            continue

        try:
            result = predict_steel(image_path)
            defect_summary = result.get("defect_summary", {})

            for c in corrections:
                original = c.get("original_class", "")
                corrected = c.get("corrected_class", "")
                action = c.get("action", "")

                key = f"class_{original}"
                if action == "remove":
                    # Check if the defect was reduced/removed
                    if key in defect_summary and not defect_summary[key]["detected"]:
                        correct += 1
                elif action == "reclassify":
                    # Check if the model now predicts the corrected class in that region
                    corrected_key = f"class_{corrected}"
                    if corrected_key in defect_summary and defect_summary[corrected_key]["detected"]:
                        correct += 1
                total += 1

        except Exception as e:
            print(f"[CaneNexus] Steel validation error for {image_path}: {e}")
            continue

    if total == 0:
        return {"val_accuracy": 0.0, "val_loss": 0.0, "correct": 0, "total": 0}

    return {
        "val_accuracy": round(correct / total, 4),
        "val_loss": 0.0,  # No differentiable loss for region-level validation
        "correct": correct,
        "total": total
    }
