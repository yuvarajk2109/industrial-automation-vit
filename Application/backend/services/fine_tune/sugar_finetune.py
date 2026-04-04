"""
CaneNexus – Sugar Fine-Tuning
Fine-tunes the sugar_head (classification) layer of DDA-ViT
using accumulated operator corrections.

Only the sugar_head (nn.Linear(256, 4)) is unfrozen by default.
Optionally, proj_q can also be unfrozen for deeper adaptation.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from config import SUGAR_CLASSES, SUGAR_IMAGE_SIZE, FINETUNE_DEFAULTS
from inference.image_utils import load_image
from services.fine_tune.validator import split_corrections, validate_sugar


class CorrectionDataset(Dataset):
    """Dataset built from operator corrections."""

    def __init__(self, corrections: list, class_to_idx: dict, image_size: int):
        self.corrections = corrections
        self.class_to_idx = class_to_idx
        self.image_size = image_size

    def __len__(self):
        return len(self.corrections)

    def __getitem__(self, idx):
        corr = self.corrections[idx]
        image_path = corr["image_path"]
        target_class = corr["corrected_label"]["class"]
        target_idx = self.class_to_idx[target_class]

        _, img_tensor, _ = load_image(image_path, size=self.image_size)
        return img_tensor.squeeze(0), target_idx


def finetune_sugar(model, corrections: list, config: dict, device,
                   progress_callback=None) -> dict:
    """
    Fine-tune the sugar classification head using accumulated corrections.

    Args:
        model: The DDA-ViT model instance.
        corrections: List of feedback documents with corrected_label.class.
        config: Hyperparameters dict (lr, epochs, min_corrections, etc.).
        device: Torch device.
        progress_callback: Optional callable(epoch, total_epochs, metrics)
            for reporting progress.

    Returns:
        Dict with training metrics and the updated state dict.
        {
            "train_loss": float,
            "val_loss": float,
            "val_accuracy": float,
            "epochs_run": int,
            "state_dict": OrderedDict (model state)
        }

    Raises:
        ValueError if insufficient corrections.
    """
    # ── Config with defaults ──
    lr = config.get("lr", FINETUNE_DEFAULTS["lr"])
    epochs = config.get("epochs", FINETUNE_DEFAULTS["epochs"])
    min_corrections = config.get("min_corrections", FINETUNE_DEFAULTS["min_corrections"])
    val_split = config.get("validation_split", FINETUNE_DEFAULTS["validation_split"])
    unfreeze_proj = config.get("unfreeze_projection", FINETUNE_DEFAULTS["unfreeze_projection"])
    patience = config.get("early_stopping_patience", FINETUNE_DEFAULTS["early_stopping_patience"])

    if len(corrections) < min_corrections:
        raise ValueError(
            f"Need at least {min_corrections} corrections, got {len(corrections)}."
        )

    # ── Split into train/val ──
    train_corr, val_corr = split_corrections(corrections, val_split)

    print(f"[CaneNexus] Sugar fine-tune: {len(train_corr)} train, {len(val_corr)} val")

    # ── Freeze everything ──
    for param in model.parameters():
        param.requires_grad = False

    # ── Unfreeze sugar_head ──
    for param in model.sugar_head.parameters():
        param.requires_grad = True

    # ── Optionally unfreeze proj_q ──
    if unfreeze_proj:
        for param in model.proj_q.parameters():
            param.requires_grad = True

    # ── Build dataset ──
    class_to_idx = {cls: i for i, cls in enumerate(SUGAR_CLASSES)}
    dataset = CorrectionDataset(train_corr, class_to_idx, SUGAR_IMAGE_SIZE)
    loader = DataLoader(dataset, batch_size=min(len(train_corr), 8), shuffle=True)

    # ── Optimizer ──
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # ── Training loop ──
    model.train()
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    metrics_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            output = model(x_sugar=images)
            loss = F.cross_entropy(output, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ──
        model.eval()
        val_metrics = validate_sugar(model, val_corr, device)
        model.train()

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 6),
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["val_accuracy"]
        }
        metrics_history.append(epoch_metrics)

        print(
            f"[CaneNexus] Epoch {epoch + 1}/{epochs} — "
            f"train_loss: {avg_train_loss:.6f}, "
            f"val_loss: {val_metrics['val_loss']:.6f}, "
            f"val_acc: {val_metrics['val_accuracy']:.4f}"
        )

        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_metrics)

        # ── Early stopping ──
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
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

    # Final metrics = last epoch's metrics (or best)
    final_metrics = metrics_history[-1] if metrics_history else {}

    return {
        "train_loss": final_metrics.get("train_loss", 0),
        "val_loss": final_metrics.get("val_loss", 0),
        "val_accuracy": final_metrics.get("val_accuracy", 0),
        "epochs_run": len(metrics_history),
        "metrics_history": metrics_history,
        "state_dict": model.state_dict()
    }
