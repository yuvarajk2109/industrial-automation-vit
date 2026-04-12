"""
Singleton Model Loader
    - Loads the DDA-ViT model
    - provides it to all consumers
"""

import torch
import segmentation_models_pytorch as smp
import timm

from config import (
    STEEL_MODEL_PATH, SUGAR_MODEL_PATH,
    NUM_STEEL_CLASSES, NUM_SUGAR_CLASSES
)
from models.dda_vit import DDAViT

# Device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level singleton
_model = None


def _freeze_model(model):
    """
    - Freezes all parameters of a model
    - Used during inference and finetuning
    """
    for p in model.parameters():
        p.requires_grad = False


def _load_model():
    """
    - Loads steel + sugar base models
    - Gracefully handles whether
        - checkpoints on disk are 
            - raw baseline architectures (OR)
            - localized DDA-ViT
    - Component state dictionaries produced by domain-specific fine-tuning
    """
    global _model

    print(f"[CaneNexus] Loading models on device: {device}")
    
    try:
        steel_state = torch.load(str(STEEL_MODEL_PATH), map_location=device)
    except Exception:
        print("[CaneNexus] WARNING: Failed to load steel.pth")
        steel_state = {}
        
    try:
        sugar_state = torch.load(str(SUGAR_MODEL_PATH), map_location=device)
    except Exception:
        print("[CaneNexus] WARNING: Failed to load sugar.pth")
        sugar_state = {}

    steel_is_dda = any("steel.encoder." in k for k in steel_state.keys())
    sugar_is_dda = any("sugar.model." in k for k in sugar_state.keys())

    steel_model = smp.Unet(
        encoder_name="mit_b4",
        encoder_weights=None,
        in_channels=3,
        classes=NUM_STEEL_CLASSES
    )
    steel_mapped = {k.replace("steel.", ""): v for k, v in steel_state.items() if k.startswith("steel.")} if steel_is_dda else steel_state
    steel_model.load_state_dict(steel_mapped, strict=False)

    sugar_model = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        num_classes=NUM_SUGAR_CLASSES
    )
    sugar_mapped = {k.replace("sugar.model.", ""): v for k, v in sugar_state.items() if k.startswith("sugar.model.")} if sugar_is_dda else sugar_state
    sugar_model.load_state_dict(sugar_mapped, strict=False)

    _freeze_model(steel_model)
    _freeze_model(sugar_model)

    model = DDAViT(steel_model, sugar_model)
    
    # Inject Domain-Specific Heads & Projections
    final_state = model.state_dict()
    
    if steel_is_dda:
        for k, v in steel_state.items():
            if k.startswith("proj_s.") or k.startswith("seg_head."):
                final_state[k] = v
                
    if sugar_is_dda:
        for k, v in sugar_state.items():
            if k.startswith("proj_q.") or k.startswith("sugar_head.") or k.startswith("cross_attn."):
                final_state[k] = v

    model.load_state_dict(final_state, strict=True)
    model.to(device)
    model.eval()

    _model = model
    print("[CaneNexus] DDA-ViT model loaded successfully.")
    return _model


def get_model():
    """
    - Gets or lazily initialises the DDA-ViT model singleton
    """
    global _model
    if _model is None:
        _load_model()
    return _model


def get_device():
    """
    - Return the active torch device
    - Preferably use CUDA (GPU)
    """
    return device