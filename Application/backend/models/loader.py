"""
CaneNexus – Singleton Model Loader
Loads the DDA-ViT model once and provides it to all consumers.
"""

import torch
import segmentation_models_pytorch as smp
import timm

from config import (
    STEEL_MODEL_PATH, SUGAR_MODEL_PATH,
    NUM_STEEL_CLASSES, NUM_SUGAR_CLASSES
)
from models.dda_vit import DDAViT

# ── Device ──
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Module-level singleton ──
_model = None


def _freeze_model(model):
    """Freeze all parameters of a model."""
    for p in model.parameters():
        p.requires_grad = False


def _load_model():
    """
    Load steel + sugar base models, wrap them in DDAViT,
    freeze backbones, and set to eval mode.
    """
    global _model

    print(f"[CaneNexus] Loading models on device: {device}")
    print(f"[CaneNexus] Steel model path: {STEEL_MODEL_PATH}")
    print(f"[CaneNexus] Sugar model path: {SUGAR_MODEL_PATH}")

    # ── Steel Model (SegFormer UNet with mit_b4 encoder) ──
    steel_model = smp.Unet(
        encoder_name="mit_b4",
        encoder_weights=None,       # trained weights loaded manually
        in_channels=3,
        classes=NUM_STEEL_CLASSES
    )
    steel_model.load_state_dict(
        torch.load(str(STEEL_MODEL_PATH), map_location=device)
    )
    steel_model.to(device)
    steel_model.eval()

    # ── Sugar Model (Swin Tiny via timm) ──
    sugar_model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,           # avoid overwriting trained weights
        num_classes=NUM_SUGAR_CLASSES
    )
    sugar_model.load_state_dict(
        torch.load(str(SUGAR_MODEL_PATH), map_location=device)
    )
    sugar_model.to(device)
    sugar_model.eval()

    # ── Freeze both base models ──
    _freeze_model(steel_model)
    _freeze_model(sugar_model)

    # ── Wrap in DDA-ViT ──
    model = DDAViT(steel_model, sugar_model)
    model.to(device)
    model.eval()

    _model = model
    print("[CaneNexus] DDA-ViT model loaded successfully.")
    return _model


def get_model():
    """Get or lazily initialise the DDA-ViT model singleton."""
    global _model
    if _model is None:
        _load_model()
    return _model


def get_device():
    """Return the active torch device."""
    return device
