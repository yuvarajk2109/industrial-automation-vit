import os
import csv
import torch
import random
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

import timm

random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..\\\\.."))
DATA_DIR = os.path.join(ROOT_DIR, "Datasets", "sugar-quality-inspection")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")

print(f"""
Train Images Directory: {(TRAIN_IMAGES_DIR, os.path.exists(TRAIN_IMAGES_DIR))}
Test Images Directory: {(TEST_IMAGES_DIR, os.path.exists(TEST_IMAGES_DIR))}
"""
)

IMAGE_SIZE = 224
BATCH_SIZE = 32

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

val_test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
])

NUM_CLASSES = 4
MAX_EPOCHS = 50
PATIENCE = 7

MODEL_NAME = "swin_tiny_patch4_window7_224"

BEST_MODEL_DIRECTORY = os.path.join(ROOT_DIR, "Models", "Sugar", MODEL_NAME)
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
RESULTS_FILE = os.path.join(RESULTS_DIR, "model_results_sugar.csv")
print("Results file path:", RESULTS_FILE)

RESULT_METRICS_DIR = os.path.join(ROOT_DIR, "Results", "Sugar", MODEL_NAME)
os.makedirs(RESULT_METRICS_DIR, exist_ok=True)
print("Result metrics directory:", RESULT_METRICS_DIR)

STATE_TEMPLATES = {
    "unsaturated": {
        "supersaturation_ratio": (0.95, 1.00),
        "nucleation_risk": "none",
        "growth_stability": "none"
    },
    "metastable": {
        "supersaturation_ratio": (1.05, 1.25),
        "nucleation_risk": "low",
        "growth_stability": "stable"
    },
    "intermediate": {
        "supersaturation_ratio": (1.20, 1.35),
        "nucleation_risk": "medium",
        "growth_stability": "transition"
    },
    "labile": {
        "supersaturation_ratio": (1.30, 1.50),
        "nucleation_risk": "high",
        "growth_stability": "unstable"
    }
}

def infer_supersaturation_range(crystal_state, confidence):
    low, high = STATE_TEMPLATES[crystal_state]["supersaturation_ratio"]
    center = (low + high) / 2
    width = (high - low) * (1 - confidence)
    return (center - width / 2, center + width / 2)

def build_inferred_state(vit_output):
    cs = vit_output["crystal_state"]
    conf = vit_output["confidence"]

    inferred = {
        "crystal_state": cs,
        "confidence": conf,
        "supersaturation_ratio_range": infer_supersaturation_range(cs, conf),
        "nucleation_risk": STATE_TEMPLATES[cs]["nucleation_risk"],
        "growth_stability": STATE_TEMPLATES[cs]["growth_stability"]
    }

    return inferred

CDG = nx.DiGraph()

nodes = {
    "observed": ["crystal_state"],
    "derived": [
        "supersaturation_ratio_range",
        "nucleation_risk",
        "growth_stability"
    ],
    "action": [
        "reduce_seeding",
        "emergency_dilution",
        "hold_process",
        "increase_evaporation"
    ]
}

for layer, ns in nodes.items():
    for n in ns:
        CDG.add_node(n, layer=layer)

CDG.add_edge("crystal_state", "supersaturation_ratio_range")
CDG.add_edge("crystal_state", "nucleation_risk")
CDG.add_edge("crystal_state", "growth_stability")

CDG.add_edge("nucleation_risk", "reduce_seeding")
CDG.add_edge("nucleation_risk", "emergency_dilution")
CDG.add_edge("growth_stability", "hold_process")
CDG.add_edge("growth_stability", "increase_evaporation")


def decide_actions(derived_state):
    actions = []

    if derived_state["nucleation_risk"] == "high":
        actions.append("reduce_seeding")
        actions.append("emergency_dilution")

    if derived_state["growth_stability"] == "stable":
        actions.append("hold_process")
    else:
        actions.append("increase_evaporation")

    return actions

pos = nx.multipartite_layout(CDG, subset_key="layer")

color_map = []
for n in CDG.nodes():
    layer = CDG.nodes[n]["layer"]
    if layer == "observed":
        color_map.append("lightblue")
    elif layer == "derived":
        color_map.append("orange")
    else:
        color_map.append("salmon")

plt.figure(figsize=(12, 8))
nx.draw(
    CDG,
    pos,
    with_labels=True,
    node_color=color_map,
    node_size=2600,
    font_size=9,
    edge_color="gray"
)
plt.title("Sugar Crystallization Decision Graph")
plt.show()

vit_output = {
    "crystal_state": "labile",
    "confidence": 0.87
}

kg_input_state = build_inferred_state(vit_output)

final_actions = decide_actions(kg_input_state)

CSG = nx.DiGraph()

CSG.add_node(
    "UNSATURATED",
    sigma_range=(0.95, 1.00),
    nucleation="none",
    growth="none",
    visual="clear_liquid"
)

CSG.add_node(
    "METASTABLE",
    sigma_range=(1.05, 1.25),
    nucleation="low",
    growth="controlled",
    visual="clear_with_crystals"
)

CSG.add_node(
    "INTERMEDIATE",
    sigma_range=(1.20, 1.35),
    nucleation="medium",
    growth="mixed",
    visual="cloudy"
)

CSG.add_node(
    "LABILE",
    sigma_range=(1.30, 1.50),
    nucleation="high",
    growth="uncontrolled",
    visual="opaque_fines"
)

CSG.add_edge(
    "UNSATURATED",
    "METASTABLE",
    trigger="supersaturation_increase",
    risk="low",
    controllability="high"
)

CSG.add_edge(
    "METASTABLE",
    "INTERMEDIATE",
    trigger="excess_evaporation",
    risk="medium",
    controllability="medium"
)

CSG.add_edge(
    "INTERMEDIATE",
    "LABILE",
    trigger="runaway_supersaturation",
    risk="high",
    controllability="low"
)

CSG.add_edge(
    "LABILE",
    "INTERMEDIATE",
    trigger="dilution_or_cooling",
    risk="medium",
    controllability="medium"
)

CSG.add_edge(
    "INTERMEDIATE",
    "METASTABLE",
    trigger="stabilized_growth",
    risk="low",
    controllability="high"
)

def score_transitions(graph, active_state):
    scores = {}

    for _, dst, data in graph.out_edges(active_state, data=True):
        risk_weight = {"low": 0.2, "medium": 0.5, "high": 0.8}[data["risk"]]
        controllability = {"high": 0.9, "medium": 0.6, "low": 0.3}[data["controllability"]]

        scores[dst] = risk_weight * (1 - controllability)

    return scores

pos = nx.circular_layout(CSG)

plt.figure(figsize=(8, 8))
nx.draw(
    CSG,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=10,
    edge_color="gray"
)

edge_labels = {
    (u, v): d["trigger"]
    for u, v, d in CSG.edges(data=True)
}

nx.draw_networkx_edge_labels(CSG, pos, edge_labels=edge_labels)

plt.title("Crystallization State Transition Graph")
plt.show()

vit_output = {
    "predicted_state": "LABILE",
    "confidence": 0.87
}

for node in CSG.nodes():
    CSG.nodes[node]["belief"] = 0.0

CSG.nodes[vit_output["predicted_state"]]["belief"] = vit_output["confidence"]

transition_scores = score_transitions(CSG, "LABILE")

KG = nx.DiGraph()

states = {
    "UNSATURATED": {
        "sigma_range": (0.95, 1.00),
        "description": "No crystal growth possible"
    },
    "METASTABLE": {
        "sigma_range": (1.05, 1.25),
        "description": "Controlled crystal growth zone"
    },
    "INTERMEDIATE": {
        "sigma_range": (1.20, 1.35),
        "description": "Transition regime"
    },
    "LABILE": {
        "sigma_range": (1.30, 1.50),
        "description": "Uncontrolled nucleation regime"
    }
}

conditions = [
    "low_nucleation_risk",
    "medium_nucleation_risk",
    "high_nucleation_risk",
    "stable_growth",
    "unstable_growth"
]

actions = [
    "hold_process",
    "increase_evaporation",
    "reduce_seeding",
    "emergency_dilution"
]

for s, attrs in states.items():
    KG.add_node(s, node_type="state", **attrs)
for c in conditions:
    KG.add_node(c, node_type="condition")
for a in actions:
    KG.add_node(a, node_type="action")

KG.add_edge("UNSATURATED", "METASTABLE",
            edge_type="state_transition",
            trigger="increase_supersaturation")

KG.add_edge("METASTABLE", "INTERMEDIATE",
            edge_type="state_transition",
            trigger="excess_evaporation")

KG.add_edge("INTERMEDIATE", "LABILE",
            edge_type="state_transition",
            trigger="runaway_supersaturation")

KG.add_edge("LABILE", "INTERMEDIATE",
            edge_type="state_transition",
            trigger="cooling_or_dilution")

KG.add_edge("INTERMEDIATE", "METASTABLE",
            edge_type="state_transition",
            trigger="growth_stabilization")

KG.add_edge("UNSATURATED", "low_nucleation_risk", edge_type="implies_condition")
KG.add_edge("METASTABLE", "low_nucleation_risk", edge_type="implies_condition")
KG.add_edge("METASTABLE", "stable_growth", edge_type="implies_condition")

KG.add_edge("INTERMEDIATE", "medium_nucleation_risk", edge_type="implies_condition")
KG.add_edge("INTERMEDIATE", "unstable_growth", edge_type="implies_condition")

KG.add_edge("LABILE", "high_nucleation_risk", edge_type="implies_condition")
KG.add_edge("LABILE", "unstable_growth", edge_type="implies_condition")

KG.add_edge("low_nucleation_risk", "hold_process",
            edge_type="triggers_action")

KG.add_edge("stable_growth", "hold_process",
            edge_type="triggers_action")

KG.add_edge("unstable_growth", "increase_evaporation",
            edge_type="triggers_action")

KG.add_edge("high_nucleation_risk", "reduce_seeding",
            edge_type="triggers_action")

KG.add_edge("high_nucleation_risk", "emergency_dilution",
            edge_type="triggers_action")

def infer_actions(graph, active_state):
    actions = set()

    for _, cond in graph.out_edges(active_state):
        if graph.nodes[cond]["node_type"] == "condition":
            for _, act in graph.out_edges(cond):
                if graph.nodes[act]["node_type"] == "action":
                    actions.add(act)

    return list(actions)

vit_output = {
    "state": "LABILE",
    "confidence": 0.87
}

for n in KG.nodes():
    KG.nodes[n]["belief"] = 0.0

KG.nodes[vit_output["state"]]["belief"] = vit_output["confidence"]

final_actions = infer_actions(KG, vit_output["state"])
final_actions

pos = nx.spring_layout(KG, seed=42)

color_map = []
for n in KG.nodes():
    t = KG.nodes[n]["node_type"]
    if t == "state":
        color_map.append("lightblue")
    elif t == "condition":
        color_map.append("orange")
    else:
        color_map.append("salmon")

plt.figure(figsize=(14, 10))
nx.draw(
    KG,
    pos,
    with_labels=True,
    node_color=color_map,
    node_size=2800,
    font_size=9,
    edge_color="gray"
)

edge_labels = {
    (u, v): d["edge_type"]
    for u, v, d in KG.edges(data=True)
}

nx.draw_networkx_edge_labels(KG, pos, edge_labels=edge_labels, font_size=7)

plt.title("Unified Sugar Crystallization Knowledge Graph")
plt.show()

def layered_layout(graph):
    pos = {}

    x_pos = {
        "state": 0.0,
        "condition": 1.5,
        "action": 3.0
    }

    layers = {"state": [], "condition": [], "action": []}
    for n, d in graph.nodes(data=True):
        layers[d["node_type"]].append(n)

    for layer, nodes in layers.items():
        y_spacing = 1.0
        start_y = -(len(nodes) - 1) / 2

        for i, node in enumerate(sorted(nodes)):
            pos[node] = (x_pos[layer], start_y + i * y_spacing)

    return pos

pos = layered_layout(KG)

plt.figure(figsize=(18, 11))

node_colors = []
node_sizes = []

for n in KG.nodes():
    t = KG.nodes[n]["node_type"]
    if t == "state":
        node_colors.append("#9ecae1") 
        node_sizes.append(3400)
    elif t == "condition":
        node_colors.append("#fdae6b")
        node_sizes.append(3000)
    else:
        node_colors.append("#fc9272")
        node_sizes.append(3200)

nx.draw_networkx_nodes(
    KG,
    pos,
    node_color=node_colors,
    node_size=node_sizes,
    edgecolors="black",
    linewidths=1.2
)

nx.draw_networkx_edges(
    KG,
    pos,
    arrowstyle="->",
    arrowsize=18,
    edge_color="#555555",
    width=1.6,
    connectionstyle="arc3,rad=0.15"
)

nx.draw_networkx_labels(
    KG,
    pos,
    font_size=10,
    font_weight="bold"
)

edge_labels = {
    (u, v): d["edge_type"]
    for u, v, d in KG.edges(data=True)
}

nx.draw_networkx_edge_labels(
    KG,
    pos,
    edge_labels=edge_labels,
    font_size=8,
    label_pos=0.55,
    rotate=False
)

plt.title(
    "Unified Sugar Crystallization Knowledge Graph\n"
    "(Vision State → Inferred Condition → Control Action)",
    fontsize=14,
    pad=25
)


legend_elements = [
    Patch(facecolor="#9ecae1", edgecolor="black",
          label="Crystallization State (ViT Output)"),
    Patch(facecolor="#fdae6b", edgecolor="black",
          label="Inferred Process Condition"),
    Patch(facecolor="#fc9272", edgecolor="black",
          label="Control / Decision Action")
]

plt.legend(
    handles=legend_elements,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
    frameon=False,
    fontsize=10
)

plt.axis("off")
plt.tight_layout()

## Save the graph visualization to Media (as Sugar_Crystallization_KG.png)
output_path = os.path.join(ROOT_DIR, "Media", "Sugar_Crystallization_KG.png")
plt.savefig(output_path)

plt.show()

full_train_dataset = datasets.ImageFolder(
    root=TRAIN_IMAGES_DIR,
    transform=train_transforms
)

class_names = full_train_dataset.classes
NUM_CLASSES = len(class_names)

print("Classes:", class_names)
print("Total train images:", len(full_train_dataset))
print("Number of classes:", NUM_CLASSES)

test_dataset = datasets.ImageFolder(
    root=TEST_IMAGES_DIR,
    transform=val_test_transforms
)

print("Total test images:", len(test_dataset))

total_size = len(full_train_dataset)

train_size = int(0.70 * total_size)
val_size   = total_size - train_size

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(random_seed)
)

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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES
)

model = model.to(device)

def print_classifier_head(model):
    if hasattr(model, "head"):
        print("Classifier head:", model.head)
    elif hasattr(model, "fc"):
        print("Classifier head:", model.fc)
    elif hasattr(model, "classifier"):
        print("Classifier head:", model.classifier)
    else:
        print("Classifier head not found")

print_classifier_head(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4 # weight decay --> improves regularization!
)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(loader)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

best_val_acc = 0.0
epochs_without_improvement = 0

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

print(
    f"{'Epoch':<8}"
    f"{'Train Loss':<15}"
    f"{'Val Loss':<15}"
    f"{'Train Acc':<12}"
    f"{'Val Acc':<12}"
    f"{'Status':<20}"
)

print("-" * 85)

for epoch in range(1, MAX_EPOCHS + 1):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        optimizer,
        criterion,
        device
    )

    val_loss, val_acc = validate_one_epoch(
        model,
        val_loader,
        criterion,
        device
    )

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    if val_acc > best_val_acc:

        best_val_acc = val_acc
        epochs_without_improvement = 0

        torch.save(model.state_dict(), BEST_MODEL_PATH)

        status = "New Best Model"

    else:

        epochs_without_improvement += 1
        status = f"No Improvement ({epochs_without_improvement}/{PATIENCE})"


    # ---------- Tabular logging ----------
    print(
        f"{epoch:<8}"
        f"{train_loss:<15.4f}"
        f"{val_loss:<15.4f}"
        f"{train_acc:<12.4f}"
        f"{val_acc:<12.4f}"
        f"{status:<20}"
    )


    # ---------- Early stopping ----------
    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        break

# Load best model for testing
model.load_state_dict(torch.load(BEST_MODEL_PATH))
model.eval()

print("Accuracy = {:.4f}".format(best_val_acc))

epochs_trained = len(train_losses)
epochs_range = range(1, epochs_trained + 1)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(epochs_range, train_accuracies, label="Train Acc")
plt.plot(epochs_range, val_accuracies, label="Val Acc")
plt.legend()
plt.title("Accuracy Curve")

plt.show()

all_preds = []
all_labels = []

model.eval()

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
print("Test Accuracy: {:.2f}%".format(accuracy))

cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

output_path = os.path.join(RESULT_METRICS_DIR, "confusion_matrix.png")
plt.savefig(output_path)

plt.show()

report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names,
    digits=4
)

print("Classification Report:\n")
print(report)

# Write report to .txt file in RESULT_METRICS_DIR with name classification_report.txt
report_path = os.path.join(RESULT_METRICS_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)

row = [
    MODEL_NAME,
    round(best_val_acc, 5),
    round(val_losses[val_accuracies.index(best_val_acc)], 5),
    round(accuracy, 2),
    epochs_trained
]

file_exists = os.path.isfile(RESULTS_FILE)

with open(RESULTS_FILE, mode="a", newline="") as f:
    writer = csv.writer(f)

    # Write header only once
    if not file_exists:
        writer.writerow([
            "model_name",
            "best_val_accuracy",
            "best_val_loss",
            "test_accuracy",
            "epochs_trained"
        ])

    writer.writerow(row)

print(f"Results logged successfully to {RESULTS_FILE}.")
print("Logged row:", row)