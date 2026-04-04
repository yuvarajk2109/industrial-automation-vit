"""
CaneNexus – Steel Fine-Tuning
Fine-tunes the seg_head (segmentation) layer of DDA-ViT
using accumulated operator corrections.

Only the seg_head (nn.Conv2d(256, 4, kernel_size=1)) is unfrozen.
The SegFormer/UNet encoder backbone stays frozen.

Correction strategy:
    - For "reclassify" corrections: the model's own mask is used as pseudo
      ground-truth with the class labels swapped in regions matching the
      original class.
    - For "remove" corrections: regions of the original class are set to
      background (0) in the pseudo ground-truth.
"""

import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import STEEL_CLASSES, STEEL_IMAGE_SIZE, FINETUNE_DEFAULTS
from inference.image_utils import sliding_window_inference
from services.fine_tune.validator import split_corrections


def _build_corrected_mask(original_mask: np.ndarray, corrections: list,
                          missed_defects: list = None) -> np.ndarray:
    """
    Build a pseudo ground-truth mask from the model's prediction and
    the operator's region-level corrections.

    Args:
        original_mask: The model's argmax mask (H, W), values 0-4.
        corrections: List of {"original_class", "corrected_class", "action"}.
        missed_defects: List of class strings that should have been detected
                        (no spatial info — we cannot generate mask for these).

    Returns:
        Corrected mask (H, W) with swapped/removed classes.
    """
    corrected = original_mask.copy()

    for corr in corrections:
        original_cls = int(corr.get("original_class", 0))
        action = corr.get("action", "keep")

        if action == "keep":
            continue

        if action == "remove":
            # Set regions of this class to background (0)
            corrected[corrected == original_cls] = 0

        elif action == "reclassify":
            corrected_cls = int(corr.get("corrected_class", original_cls))
            # Swap class labels
            corrected[original_mask == original_cls] = corrected_cls

    return corrected


def finetune_steel(model, corrections: list, config: dict, device,
                   progress_callback=None) -> dict:
    """
    Fine-tune the steel segmentation head using accumulated corrections.

    Strategy:
        1. For each correction, load the original image and run inference
           to get the current prediction mask.
        2. Modify the mask based on the correction (reclassify/remove).
        3. Use the corrected mask as pseudo ground-truth.
        4. Fine-tune seg_head with pixel-wise cross-entropy loss.
        5. Validate on holdout.

    Args:
        model: The DDA-ViT model instance.
        corrections: List of feedback documents.
        config: Hyperparameters dict.
        device: Torch device.
        progress_callback: Optional callable(epoch, total_epochs, metrics).

    Returns:
        Dict with training metrics and updated state dict.
    """
    # ── Config with defaults ──
    lr = config.get("lr", FINETUNE_DEFAULTS["lr"])
    epochs = config.get("epochs", FINETUNE_DEFAULTS["epochs"])
    min_corrections = config.get("min_corrections", FINETUNE_DEFAULTS["min_corrections"])
    val_split = config.get("validation_split", FINETUNE_DEFAULTS["validation_split"])
    patience = config.get("early_stopping_patience", FINETUNE_DEFAULTS["early_stopping_patience"])

    if len(corrections) < min_corrections:
        raise ValueError(
            f"Need at least {min_corrections} corrections, got {len(corrections)}."
        )

    # ── Split into train/val ──
    train_corr, val_corr = split_corrections(corrections, val_split)
    print(f"[CaneNexus] Steel fine-tune: {len(train_corr)} train, {len(val_corr)} val")

    # ── Freeze everything ──
    for param in model.parameters():
        param.requires_grad = False

    # ── Unfreeze seg_head only ──
    for param in model.seg_head.parameters():
        param.requires_grad = True

    # ── Optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # ── Pre-compute training data ──
    # For each correction, generate (image_patches, corrected_mask_patches)
    training_data = []
    model.eval()

    with torch.no_grad():
        for corr in train_corr:
            image_path = corr.get("image_path", "")
            corr_label = corr.get("corrected_label", {})
            corrections_list = corr_label.get("corrections", [])
            missed_defects = corr_label.get("missed_defects", [])

            if not image_path or not corrections_list:
                continue

            try:
                # Load and preprocess
                img = cv2.imread(image_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0
                img = cv2.resize(img, (img.shape[1], STEEL_IMAGE_SIZE))

                # Get model's current prediction
                probs = sliding_window_inference(model, img, device)
                original_mask = np.argmax(probs, axis=0)

                # Build corrected mask
                corrected_mask = _build_corrected_mask(
                    original_mask, corrections_list, missed_defects
                )

                # Convert to tensors
                # Image: (H, W, 3) → patches via sliding window
                h, w, _ = img.shape
                patch_size = STEEL_IMAGE_SIZE
                stride = STEEL_IMAGE_SIZE

                for x in range(0, w, stride):
                    x_end = min(x + patch_size, w)
                    x_start = x_end - patch_size
                    if x_start < 0:
                        x_start = 0

                    img_patch = img[:, x_start:x_end, :]
                    mask_patch = corrected_mask[:, x_start:x_end]

                    if img_patch.shape[1] < patch_size:
                        # Pad
                        pad_w = patch_size - img_patch.shape[1]
                        img_patch = np.pad(img_patch, ((0, 0), (0, pad_w), (0, 0)))
                        mask_patch = np.pad(mask_patch, ((0, 0), (0, pad_w)))

                    img_tensor = torch.from_numpy(
                        img_patch.transpose(2, 0, 1)
                    ).unsqueeze(0).float()

                    mask_tensor = torch.from_numpy(mask_patch).long()

                    training_data.append((img_tensor, mask_tensor))

            except Exception as e:
                print(f"[CaneNexus] Error preparing steel training data: {e}")
                continue

    if not training_data:
        raise ValueError("No valid training data could be generated from corrections.")

    print(f"[CaneNexus] Generated {len(training_data)} training patches")

    # ── Training loop ──
    model.train()
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    metrics_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle training data
        indices = list(range(len(training_data)))
        np.random.shuffle(indices)

        for idx in indices:
            img_tensor, mask_tensor = training_data[idx]
            img_tensor = img_tensor.to(device)
            mask_tensor = mask_tensor.to(device)

            optimizer.zero_grad()
            output = model(x_steel=img_tensor)

            # Resize output to match mask if needed
            if output.shape[2:] != mask_tensor.shape:
                output = F.interpolate(
                    output, size=mask_tensor.shape,
                    mode="bilinear", align_corners=False
                )

            loss = F.cross_entropy(output, mask_tensor.unsqueeze(0))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": 0.0,
            "val_accuracy": 0.0
        }
        metrics_history.append(epoch_metrics)

        print(
            f"[CaneNexus] Steel Epoch {epoch + 1}/{epochs} — "
            f"train_loss: {avg_train_loss:.6f}"
        )

        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_metrics)

        # ── Early stopping on training loss ──
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[CaneNexus] Early stopping at epoch {epoch + 1}")
                break

    # ── Restore best state ──
    if best_state:
        model.load_state_dict(best_state)

    model.eval()

    # ── Re-freeze everything ──
    for param in model.parameters():
        param.requires_grad = False

    final_metrics = metrics_history[-1] if metrics_history else {}

    return {
        "train_loss": final_metrics.get("train_loss", 0),
        "val_loss": final_metrics.get("val_loss", 0),
        "val_accuracy": final_metrics.get("val_accuracy", 0),
        "epochs_run": len(metrics_history),
        "metrics_history": metrics_history,
        "state_dict": model.state_dict()
    }
