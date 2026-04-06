# # Dual Domain Adaptive ViT

# - A cross-domain feature alignment module
# - Enables interaction between steel and sugar representations
# - Projects features from each domain into a shared latent space  
# - Applies attention-based mechanisms to capture complementary structural and textural relationships. 

# *Why?*

# - Facilitates knowledge transfer across tasks; each branch benefits from patterns learned in the other domain
# - Doesn't compromise domain-specific specialization
# - Useful in industrial settings where different inspection tasks share underlying visual characteristics, 
# - Improves generalization
# - Enhances robustness to variations
# - Reduces need for large domain-specific datasets by leveraging shared feature understanding


# Modules
import os
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Times New Roman'})
plt.rcParams.update({'font.size': 14})

import torch
import torch.nn as nn
import torch.nn.functional as F

import segmentation_models_pytorch as smp
import timm

# Configs
BASE_DIR = Path().resolve()
ROOT_DIR = (BASE_DIR.parent).parent

STEEL_MODEL_PATH = os.path.join(ROOT_DIR,  "Models", "Final", "steel.pth")
SUGAR_MODEL_PATH = os.path.join(ROOT_DIR, "Models", "Final", "sugar.pth")

# Print both paths neatly, check if they exist
print(f"Steel model path: {STEEL_MODEL_PATH}")
print(f"Sugar model path: {SUGAR_MODEL_PATH}")
print(f"Steel model exists: {os.path.exists(STEEL_MODEL_PATH)}")
print(f"Sugar model exists: {os.path.exists(SUGAR_MODEL_PATH)}")

STEEL_IMAGE_PATH = os.path.join(ROOT_DIR, "Datasets", "steel-defect-detection", "test_images", "00a0b7730.jpg")
SUGAR_IMAGE_PATH = os.path.join(ROOT_DIR, "Datasets", "sugar-quality-inspection", "test_images", "metastable", "1.jpg")

# Print both paths neatly, check if they exist
print(f"Steel image path: {STEEL_IMAGE_PATH}")
print(f"Sugar image path: {SUGAR_IMAGE_PATH}")
print(f"Steel image exists: {os.path.exists(STEEL_IMAGE_PATH)}")
print(f"Sugar image exists: {os.path.exists(SUGAR_IMAGE_PATH)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Classes
steel_classes = ["1", "2", "3", "4"]
sugar_classes = ["unsaturated", "metastable", "intermediate", "labile"]

NUM_STEEL_CLASSES = len(steel_classes)
NUM_SUGAR_CLASSES = len(sugar_classes)

# Steel Model
steel_model = smp.Unet(
    encoder_name="mit_b4",
    encoder_weights=None,   # set to None as trained weights are loaded
    in_channels=3,
    classes=NUM_STEEL_CLASSES
)

steel_model.load_state_dict(torch.load(STEEL_MODEL_PATH, map_location=device))
steel_model.to(device)
steel_model.eval()

# Sugar Model
sugar_model = timm.create_model(
    "swin_tiny_patch4_window7_224",
    pretrained=False,   # avoid overwriting trained weights
    num_classes=NUM_SUGAR_CLASSES
)

sugar_model.load_state_dict(torch.load(SUGAR_MODEL_PATH, map_location=device))
sugar_model.to(device)
sugar_model.eval()

x = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    out = sugar_model(x)

print("Sugar model OK:", out.shape)

x = torch.randn(1, 3, 256, 256).to(device)

with torch.no_grad():
    out = steel_model(x)

print("Steel model OK:", out.shape)

# Steel Backbone: SegFormer encoder
class SteelBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder  # SegFormer encoder

    def forward(self, x):
        features = self.encoder(x)  # list of feature maps
        return features[-1]         # deepest feature map
    
# Sugar Backbone: Swin Transformer backbone
class SugarBackbone(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        features = self.model.forward_features(x)  # supported by timm
        return features
    
# Feature Projector
class FeatureProjector(nn.Module):
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    
# Cross-Domain Feature Aligner  
class CrossDomainAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, Fs, Fq):
        out, _ = self.attn(Fs, Fq, Fq)
        return out
    
# DDA-ViT Model
class DDAViT(nn.Module):
    def __init__(self, steel_model, sugar_model, embed_dim=256,
                 num_defect_classes=4, num_quality_classes=4):
        super().__init__()

        self.steel = SteelBackbone(steel_model)
        self.sugar = SugarBackbone(sugar_model)

        self.proj_s = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.proj_q = nn.Linear(768, embed_dim)

        self.cross_attn = CrossDomainAttention(embed_dim)

        self.seg_head = nn.Conv2d(embed_dim, num_defect_classes, kernel_size=1)
        self.sugar_head = nn.Linear(embed_dim, num_quality_classes)

    def forward(self, x_steel=None, x_sugar=None):

        Fs_map, Fq = None, None

        if x_steel is not None:
            Fs_map = self.steel(x_steel)                        # (B, C, H, W)
            Fs_map = self.proj_s(Fs_map)                        # (B, d, H, W)

            B, d, H, W = Fs_map.shape
            Fs = Fs_map.flatten(2).transpose(1, 2)              # (B, N, d)

        if x_sugar is not None:
            Fq = self.sugar(x_sugar)

            if Fq.dim() == 4:                                   # (B, H, W, C)
                B, H, W, C = Fq.shape
                Fq = Fq.view(B, H * W, C)

            if Fq.dim() == 3:                                   # (B, N, C)
                Fq = self.proj_q(Fq)
                Fq = Fq.mean(dim=1, keepdim=True)

            elif Fq.dim() == 2:                                 # (B, C)
                Fq = self.proj_q(Fq)
                Fq = Fq.unsqueeze(1)

            else:
                raise ValueError(f"Unexpected sugar feature shape: {Fq.shape}")

        if Fs_map is not None and Fq is not None:
            Fs_fused = self.cross_attn(Fs, Fq)  # (B, N, d)

            Fs_map = Fs_fused.transpose(1, 2).reshape(B, d, H, W)

        if x_steel is not None:
            # seg_out = self.seg_head(Fs_map)  # (B, 4, H, W)
            # return seg_out
            seg_out = self.seg_head(Fs_map)

            # Upsample to input size
            seg_out = F.interpolate(
                seg_out,
                size=x_steel.shape[2:],  # (H, W)
                mode="bilinear",
                align_corners=False
            )

            return seg_out

        if x_sugar is not None:
            F_out = Fq.mean(dim=1)  # (B, d)
            return self.sugar_head(F_out)

        raise ValueError("Provide x_steel or x_sugar")
    
# Freezing Model Backbones
def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

freeze_model(steel_model)
freeze_model(sugar_model)

# Initialise DDA-ViT
model = DDAViT(steel_model, sugar_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimiser
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4
)

x_steel = torch.randn(2, 3, 256, 256).to(device)
x_sugar = torch.randn(2, 3, 224, 224).to(device)

out_s = model(x_steel=x_steel)
out_q = model(x_sugar=x_sugar)

print("Steel output:", out_s.shape)
print("Sugar output:", out_q.shape)

# Load Image Function
def load_image(path, size):
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

# Steel Image Segmentation - Sliding Window Inference
def sliding_window_inference(model, image, patch_size=256, stride=256):
    model.eval()

    H, W, _ = image.shape

    full_mask = np.zeros((4, H, W))
    count_map = np.zeros((H, W))

    for x in range(0, W - patch_size + 1, stride):

        patch = image[:, x:x+patch_size]
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(x_steel=patch_tensor)
            probs = torch.sigmoid(output)[0].cpu().numpy()

        full_mask[:, :, x:x+patch_size] += probs
        count_map[:, x:x+patch_size] += 1

    # Normalize overlapping regions
    full_mask /= np.maximum(count_map, 1e-6)

    return full_mask

# Steel Image Segmentation - Inference
def predict_steel(model, image_path):
    model.eval()

    img = cv2.imread(image_path)

    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = img.copy()

    img = img.astype(np.float32) / 255.0

    # Ensure height = 256
    img = cv2.resize(img, (img.shape[1], 256))

    # -------------------------
    # SLIDING WINDOW
    # -------------------------
    probs = sliding_window_inference(model, img)

    mask = np.argmax(probs, axis=0)

    # Resize back to original
    mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    # -------------------------
    # VISUALIZATION
    # -------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="jet")
    plt.title("Sliding Window Segmentation")
    plt.axis("off")

    plt.show()

    unique, counts = np.unique(mask, return_counts=True)
    total = mask.size

    print("\nDefect Summary:")
    for u, c in zip(unique, counts):
        if u == 0:
            continue
        print(f"- Class {u}: {(c/total)*100:.2f}% area")

# Sugar Quality Inspection - Inference
def predict_sugar(model, image_path, class_names):
    model.eval()

    _, img, _ = load_image(image_path, size=224)
    img = img.to(device)

    with torch.no_grad():
        output = model(x_sugar=img)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    pred = np.argmax(probs)

    print("\nPrediction:", class_names[pred])
    print("\nProbabilities:")
    for i, p in enumerate(probs):
        print(f"{class_names[i]}: {p:.4f}")

# Prediction
predict_steel(model, STEEL_IMAGE_PATH)
predict_sugar(model, SUGAR_IMAGE_PATH, sugar_classes)