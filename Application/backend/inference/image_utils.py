"""
CaneNexus – Image Utility Functions
Loading, preprocessing, sliding window inference, and mask visualisation.
Ported from Code/DDA-ViT/DDA-ViT.py.
"""

import os
import uuid
import cv2
import numpy as np
import torch

from config import OUTPUT_DIR


def load_image(path: str, size: int):
    """
    Load an image, resize to a square of `size`, and return
    the original, the normalised tensor, and the (new_h, new_w).
    """
    img = cv2.imread(path)

    if img is None:
        raise ValueError(f"Failed to load image at path: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)

    resized = cv2.resize(img, (new_w, new_h))

    # Pad to square
    pad_img = np.zeros((size, size, 3), dtype=np.uint8)
    pad_img[:new_h, :new_w] = resized

    img_norm = pad_img.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

    return img, tensor, (new_h, new_w)


def sliding_window_inference(model, image, device, patch_size=256, stride=256):
    """
    Run steel segmentation using a sliding window across the image width.
    Image height is assumed to already be `patch_size` (256).

    Args:
        model: DDAViT model instance
        image: float32 numpy array (H, W, 3) normalised to [0, 1]
        device: torch device
        patch_size: patch width/height (256)
        stride: step size (256)

    Returns:
        full_mask: (4, H, W) probability mask
    """
    model.eval()

    H, W, _ = image.shape

    full_mask = np.zeros((4, H, W))
    count_map = np.zeros((H, W))

    for x in range(0, W - patch_size + 1, stride):
        patch = image[:, x:x + patch_size]
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x_steel=patch_tensor)
            probs = torch.sigmoid(output)[0].cpu().numpy()

        full_mask[:, :, x:x + patch_size] += probs
        count_map[:, x:x + patch_size] += 1

    # Normalise overlapping regions
    full_mask /= np.maximum(count_map, 1e-6)

    return full_mask


def generate_mask_overlay(original_img, mask, filename_prefix: str):
    """
    Generate and save a colour-coded mask overlay on the original image.

    Args:
        original_img: RGB numpy array (H, W, 3) uint8
        mask: argmax mask (H, W) with class indices 0-3
        filename_prefix: prefix for saved filenames

    Returns:
        (overlay_path, raw_mask_path) – paths to saved files
    """
    # Colour map for defect classes
    colours = {
        0: [0, 0, 0],         # No defect / background – transparent
        1: [255, 0, 0],       # Class 1 – Red
        2: [0, 255, 0],       # Class 2 – Green
        3: [0, 0, 255],       # Class 3 – Blue
        4: [255, 255, 0],     # Class 4 – Yellow (unused if 0-indexed)
    }

    # Create colour mask
    h, w = mask.shape
    colour_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, colour in colours.items():
        colour_mask[mask == cls_idx] = colour

    # Resize original to match mask if needed
    if original_img.shape[:2] != mask.shape:
        original_resized = cv2.resize(original_img, (w, h))
    else:
        original_resized = original_img

    # Blend overlay
    alpha = 0.4
    overlay = cv2.addWeighted(original_resized, 1 - alpha, colour_mask, alpha, 0)

    # Generate unique filenames
    uid = uuid.uuid4().hex[:8]
    overlay_filename = f"{filename_prefix}_{uid}_overlay.png"
    raw_mask_filename = f"{filename_prefix}_{uid}_mask.png"

    overlay_path = os.path.join(str(OUTPUT_DIR), overlay_filename)
    raw_mask_path = os.path.join(str(OUTPUT_DIR), raw_mask_filename)

    # Save (convert RGB back to BGR for OpenCV)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(raw_mask_path, colour_mask[:, :, ::-1])

    return overlay_filename, raw_mask_filename
