import os
import csv
import random
import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

import segmentation_models_pytorch as smp

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA Available:", torch.cuda.is_available())
print("Device:", device)

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..\\\\.."))
DATA_DIR = os.path.join(ROOT_DIR, "Datasets", "steel-defect-detection")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")

print(f"""
CSV Path: {(TRAIN_CSV_PATH, os.path.exists(TRAIN_CSV_PATH))}
Train Images Directory: {(TRAIN_IMAGES_DIR, os.path.exists(TRAIN_IMAGES_DIR))}
Test Images Directory: {(TEST_IMAGES_DIR, os.path.exists(TEST_IMAGES_DIR))}
"""
)

NUM_CLASSES = 4
IMG_HEIGHT = 256
IMG_WIDTH = 1600
BATCH_SIZE = 8
IMAGE_SIZE = (IMG_HEIGHT, IMG_HEIGHT)

THRESHOLD = 0.5
EPOCHS = 10
MAX_EPOCHS = 50
PATIENCE = 7
MODEL_NAME = "mit_b4"

BEST_MODEL_DIRECTORY = os.path.join(ROOT_DIR, "Models", "Steel", MODEL_NAME)
print(f"Best Model Directory: {BEST_MODEL_DIRECTORY}")
# Create directory if it doesn't exist
os.makedirs(BEST_MODEL_DIRECTORY, exist_ok=True)
print("Directory created (if it didn't exist):", BEST_MODEL_DIRECTORY)

## Fetch latest name of best model from BEST_MODEL_DIRECTORY and increment the number in the name by 1, to create new BEST_MODEL_PATH
existing_models = [f for f in os.listdir(BEST_MODEL_DIRECTORY) if f.endswith(".pth")]
if existing_models:
    existing_models.sort()
    latest_model = existing_models[-1]
    latest_number = int(latest_model.split(".")[0])
    new_number = latest_number + 1
else:
    new_number = 1

BEST_MODEL_PATH = str(new_number) + ".pth"
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIRECTORY, BEST_MODEL_PATH)
BEST_MODEL_PATH = os.path.abspath(BEST_MODEL_PATH)
print("Best model path:", BEST_MODEL_PATH)

RESULTS_DIR = os.path.join(ROOT_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "model_results_steel.csv")
print("Results file path:", RESULTS_FILE)

df = pd.read_csv(TRAIN_CSV_PATH)

all_images = df['ImageId'].unique()
all_classes = [1, 2, 3, 4]

full_index = pd.MultiIndex.from_product(
    [all_images, all_classes],
    names=['ImageId', 'ClassId']
)

df_full = (
    df.set_index(['ImageId', 'ClassId'])
      .reindex(full_index)
      .reset_index()
)

OUTPUT_CSV_PATH = os.path.join(DATA_DIR, "train_full.csv")
df_full.to_csv(OUTPUT_CSV_PATH, index=False)

def rle_decode(rle_string, height=IMG_HEIGHT, width=IMG_WIDTH):
    mask = np.zeros(height * width, dtype=np.uint8)

    if pd.isna(rle_string):
        return mask.reshape((width, height)).T

    rle = list(map(int, rle_string.split()))
    starts = rle[0::2]
    lengths = rle[1::2]

    for start, length in zip(starts, lengths):
        start -= 1
        mask[start:start + length] = 1

    return mask.reshape((width, height)).T

def build_mask_for_image(image_id, df_full):

    mask = np.zeros((NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)

    image_rows = df_full[df_full['ImageId'] == image_id]

    for _, row in image_rows.iterrows():
        class_idx = int(row['ClassId']) - 1
        mask[class_idx] = rle_decode(
            row['EncodedPixels'],
            IMG_HEIGHT,
            IMG_WIDTH
        )

    return mask

G = nx.DiGraph()
layer_input = [
    "Defect_Class_1_Detected",
    "Defect_Class_2_Detected",
    "Defect_Class_3_Detected",
    "Defect_Class_4_Detected"
]
layer_attributes = [
    "Defect_Area",
    "Defect_Length",
    "Defect_Density",
    "Defect_Distribution"
]
layer_interpretation = [
    "Isolated_Minor_Defect",
    "Localized_Severe_Defect",
    "Widespread_Defect_Pattern",
    "Critical_Structural_Defect"
]
layer_quality = [
    "Acceptable_Quality",
    "Marginal_Quality",
    "Unacceptable_Quality"
]
layer_decision = [
    "Accept_Strip",
    "Downgrade_Strip",
    "Reject_Strip",
    "Manual_Inspection_Required"
]
edges_evidence_to_attr = [
    ("Defect_Class_1_Detected", "Defect_Area", "if present"),
    ("Defect_Class_2_Detected", "Defect_Area", "if present"),
    ("Defect_Class_3_Detected", "Defect_Density", "if clustered"),
    ("Defect_Class_4_Detected", "Defect_Length", "if elongated"),

    ("Defect_Class_1_Detected", "Defect_Distribution", ""),
    ("Defect_Class_2_Detected", "Defect_Distribution", ""),
    ("Defect_Class_3_Detected", "Defect_Distribution", ""),
    ("Defect_Class_4_Detected", "Defect_Distribution", "")
]
edges_attr_to_interp = [
    ("Defect_Area", "Isolated_Minor_Defect", "area < T1"),
    ("Defect_Area", "Localized_Severe_Defect", "area ≥ T2"),
    ("Defect_Length", "Critical_Structural_Defect", "length ≥ T3"),
    ("Defect_Density", "Widespread_Defect_Pattern", "density ≥ T4"),
    ("Defect_Distribution", "Widespread_Defect_Pattern", "distributed")
]
edges_interp_to_quality = [
    ("Isolated_Minor_Defect", "Acceptable_Quality", ""),
    ("Localized_Severe_Defect", "Marginal_Quality", ""),
    ("Widespread_Defect_Pattern", "Unacceptable_Quality", ""),
    ("Critical_Structural_Defect", "Unacceptable_Quality", "")
]
edges_quality_to_decision = [
    ("Acceptable_Quality", "Accept_Strip", ""),
    ("Marginal_Quality", "Downgrade_Strip", ""),
    ("Unacceptable_Quality", "Reject_Strip", "")
]
for node in layer_input:
    G.add_node(node, layer=1)
    
for node in layer_attributes:
    G.add_node(node, layer=2)
    
for node in layer_interpretation:
    G.add_node(node, layer=3)

for node in layer_quality:
    G.add_node(node, layer=4)

for node in layer_decision:
    G.add_node(node, layer=5)

for src, dst, cond in edges_evidence_to_attr:
    G.add_edge(src, dst, condition=cond)

for src, dst, cond in edges_attr_to_interp:
    G.add_edge(src, dst, condition=cond)

for src, dst, cond in edges_interp_to_quality:
    G.add_edge(src, dst, condition=cond)

for src, dst, cond in edges_quality_to_decision:
    G.add_edge(src, dst, condition=cond)

G.add_edge(
    "Critical_Structural_Defect",
    "Manual_Inspection_Required",
    condition="safety-critical"
)

def layered_layout(G):
    pos = {}
    layer_nodes = {}

    for node, data in G.nodes(data=True):
        layer = data["layer"]
        layer_nodes.setdefault(layer, []).append(node)

    for layer, nodes in layer_nodes.items():
        for i, node in enumerate(nodes):
            pos[node] = (i, -layer)

    return pos

layer_colors = {
    1: "#AED6F1",  # Input Evidence (light blue)
    2: "#A9DFBF",  # Spatial / Severity Attributes (light green)
    3: "#F9E79F",  # Interpretation (light yellow)
    4: "#F5CBA7",  # Quality Assessment (light orange)
    5: "#F1948A"   # Decision & Action (light red)
}
nodes_by_layer = {}

for node, data in G.nodes(data=True):
    layer = data["layer"]
    nodes_by_layer.setdefault(layer, []).append(node)

pos = layered_layout(G)

plt.figure(figsize=(20, 14))

for layer, nodes in nodes_by_layer.items():
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=nodes,
        node_color=layer_colors[layer],
        node_size=3000,
        edgecolors="black",
        linewidths=1.0,
        label=f"Layer {layer}"
    )

nx.draw_networkx_edges(
    G,
    pos,
    arrows=True,
    arrowsize=15
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=9,
    font_weight="bold"
)

edge_labels = {
    (u, v): d["condition"]
    for u, v, d in G.edges(data=True)
    if d["condition"]
}

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels,
    font_size=8
)

legend_patches = [
    Patch(facecolor=layer_colors[1], edgecolor="black", label="Input Evidence"),
    Patch(facecolor=layer_colors[2], edgecolor="black", label="Spatial & Severity Attributes"),
    Patch(facecolor=layer_colors[3], edgecolor="black", label="Defect Interpretation"),
    Patch(facecolor=layer_colors[4], edgecolor="black", label="Quality Assessment"),
    Patch(facecolor=layer_colors[5], edgecolor="black", label="Decision & Action"),
]

plt.legend(
    handles=legend_patches,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08), 
    ncol=3,
    frameon=True,
    fontsize=9
)

plt.title("Steel Defect Detection - Logical Knowledge Graph", fontsize=14)
plt.axis("off")

## Save the graph visualization to Media (as Steel_Defect_Detection_KG.png)
output_path = os.path.join(ROOT_DIR, "Media", "Steel_Defect_Detection_KG.png")
plt.savefig(output_path)

plt.show()

class SteelSegmentationDataset(Dataset):
    def __init__(self, df_full, image_dir, image_size):
        self.df_full = df_full
        self.image_dir = image_dir
        self.image_ids = df_full['ImageId'].unique()
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        img_path = os.path.join(self.image_dir, image_id)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image
        image = cv2.resize(image, (self.image_size[1], self.image_size[0]))

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        # Build mask
        mask = np.zeros((4, 256, 1600), dtype=np.uint8)

        rows = self.df_full[self.df_full['ImageId'] == image_id]

        for _, row in rows.iterrows():
            class_idx = int(row['ClassId']) - 1
            decoded = rle_decode(row['EncodedPixels'])
            mask[class_idx] = decoded

        # Resize mask (nearest to preserve binary)
        resized_mask = np.zeros((4, self.image_size[0], self.image_size[1]))

        for c in range(4):
            resized_mask[c] = cv2.resize(
                mask[c],
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_NEAREST
            )

        mask_tensor = torch.from_numpy(resized_mask).float()

        return image, mask_tensor
    
full_train_dataset = SteelSegmentationDataset(
    df_full=df_full,
    image_dir=TRAIN_IMAGES_DIR,
    image_size=IMAGE_SIZE
)

print("No. of train images:", len(full_train_dataset))

train_size = int(0.7 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(random_seed)
)

print("Train samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

images, masks = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Mask batch shape:", masks.shape)

model = smp.Unet(
    encoder_name=MODEL_NAME,
    encoder_weights="imagenet",
    in_channels=3,
    classes=4
)

model = model.to(device)
print("Model moved to:", device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds, targets):

        preds = torch.sigmoid(preds)

        # flatten per batch, per class
        preds = preds.reshape(preds.size(0), preds.size(1), -1)
        targets = targets.reshape(targets.size(0), targets.size(1), -1)

        intersection = (preds * targets).sum(dim=2)
        union = preds.sum(dim=2) + targets.sum(dim=2)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        # average over classes and batch
        return 1 - dice.mean()
    
class BCEDiceLoss(nn.Module):

    def __init__(self, pos_weight=None):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.dice = DiceLoss()

    def forward(self, preds, targets):

        bce_loss = self.bce(preds, targets)
        dice_loss = self.dice(preds, targets)

        return bce_loss + dice_loss
    
def dice_score(preds, targets, threshold=THRESHOLD):

    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    preds = preds.reshape(preds.size(0), preds.size(1), -1)
    targets = targets.reshape(targets.size(0), targets.size(1), -1)

    intersection = (preds * targets).sum(dim=2)
    union = preds.sum(dim=2) + targets.sum(dim=2)

    dice = (2 * intersection) / (union + 1e-8)

    dice[union == 0] = 1.0

    return dice.mean()


def dice_per_class(preds, targets, threshold=THRESHOLD):

    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    # reshape: (B, C, H, W) → (B, C, N)
    preds = preds.reshape(preds.size(0), preds.size(1), -1)
    targets = targets.reshape(targets.size(0), targets.size(1), -1)

    intersection = (preds * targets).sum(dim=2)
    union = preds.sum(dim=2) + targets.sum(dim=2)

    dice = (2 * intersection) / (union + 1e-8)

    dice[union == 0] = 1.0

    # shape: (B, C)
    return dice

def iou_score(preds, targets, threshold=0.5):

    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection

    return intersection / (union + 1e-8)

pos_weight = torch.tensor([3.0, 3.0, 2.0, 1.0]).to(device).view(1, 4, 1, 1)

criterion = BCEDiceLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

def train_epoch(model, loader, optimizer, criterion):

    model.train()

    running_loss = 0

    for images, masks in loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)

def validate_epoch(model, loader, criterion):

    model.eval()

    running_loss = 0
    dice_total = 0
    iou_total = 0

    # NEW: per-class accumulator
    per_class_dice_total = torch.zeros(4, device=device)

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            loss = criterion(outputs, masks)
            running_loss += loss.item()

            # overall Dice
            batch_dice = dice_score(outputs, masks)
            dice_total += batch_dice

            # IoU
            iou_total += iou_score(outputs, masks)

            # NEW: per-class Dice
            dice_pc = dice_per_class(outputs, masks)   # (B, C)
            per_class_dice_total += dice_pc.mean(dim=0)  # (C,)

    avg_loss = running_loss / len(loader)
    avg_dice = dice_total / len(loader)
    avg_iou = iou_total / len(loader)

    avg_per_class_dice = per_class_dice_total / len(loader)

    return avg_loss, avg_dice, avg_iou, avg_per_class_dice

best_val_dice = 0.0
best_val_dice_pc = torch.zeros(4)

epochs_without_improvement = 0

train_losses = []
val_losses = []
val_dices = []
val_dices_per_class = []

# print(
#     f"{'Epoch':<8}"
#     f"{'Train Loss':<15}"
#     f"{'Val Loss':<15}"
#     f"{'Dice':<12}"
#     f"{'IoU':<12}"
#     f"{'Status':<20}"
# )

# print("-" * 80)

print(
    f"{'Epoch':<6}"
    f"{'Train Loss':<14}"
    f"{'Val Loss':<14}"
    f"{'Dice':<10}"
    f"{'IoU':<10}"
    f"{'C1':<8}{'C2':<8}{'C3':<8}{'C4':<8}"
    f"{'Status':<20}"
)

print("-" * 110)

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        criterion
    )

    # val_loss, val_dice, val_iou = validate_epoch(
    #     model,
    #     val_loader,
    #     criterion
    # )

    val_loss, val_dice, val_iou, val_dice_pc = validate_epoch(
        model,
        val_loader,
        criterion
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # val_dices.append(val_dice)
    val_dices.append(val_dice.item())
    val_dices_per_class.append(val_dice_pc.cpu().numpy())

    if val_dice > best_val_dice:

        best_val_dice = val_dice
        epochs_without_improvement = 0

        torch.save(model.state_dict(), BEST_MODEL_PATH)

        best_val_dice_pc = val_dice_pc.clone().detach()

        status = "New Best Model"

    else:

        epochs_without_improvement += 1
        status = f"No Improvement ({epochs_without_improvement}/{PATIENCE})"

    # print(
    #     f"{epoch:<8}"
    #     f"{train_loss:<15.4f}"
    #     f"{val_loss:<15.4f}"
    #     f"{val_dice:<12.4f}"
    #     f"{val_iou:<12.4f}"
    #     f"{status:<20}"
    # )

    c1, c2, c3, c4 = val_dice_pc.tolist()

    print(
        f"{epoch:<6}"
        f"{train_loss:<14.4f}"
        f"{val_loss:<14.4f}"
        f"{val_dice:<10.4f}"
        f"{val_iou:<10.4f}"
        f"{c1:<8.4f}{c2:<8.4f}{c3:<8.4f}{c4:<8.4f}"
        f"{status:<20}"
    )

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        print(f"Best validation Dice: {best_val_dice:.4f}")
        break

model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

print("Best Validation Dice =", best_val_dice)

epochs_trained = len(train_losses)

# Convert per-class dice to list
c1, c2, c3, c4 = best_val_dice_pc.cpu().numpy()

row = [
    MODEL_NAME,
    round(best_val_dice.item(), 5),
    round(float(c1), 5),
    round(float(c2), 5),
    round(float(c3), 5),
    round(float(c4), 5),
    epochs_trained
]

file_exists = os.path.isfile(RESULTS_FILE)

with open(RESULTS_FILE, mode="a", newline="") as f:
    writer = csv.writer(f)

    # Write header only once
    if not file_exists:
        writer.writerow([
            "model_name",
            "best_val_dice",
            "dice_c1",
            "dice_c2",
            "dice_c3",
            "dice_c4",
            "epochs_trained"
        ])

    writer.writerow(row)

print(f"Results logged successfully to {RESULTS_FILE}.")
print("Logged row:", row)