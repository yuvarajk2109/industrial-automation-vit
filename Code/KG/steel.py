"""
Steel Defect Detection - Knowledge Graph Generation
Extracted from Code/DDA-ViT/steel.py (KG section only).
Generates and visualizes the Steel Defect Detection Logical Knowledge Graph.
"""

import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.getcwd(), "..\\.."))
MEDIA_DIR = os.path.join(ROOT_DIR, "Media")
os.makedirs(MEDIA_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Graph Configs
# ---------------------------------------------------------------
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Times New Roman'})
plt.rcParams.update({'font.size': 14})
node_size = 6000

# ---------------------------------------------------------------
# Build the Knowledge Graph
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Edges
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Add nodes with layer attribute
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Add edges with condition attribute
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------
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
        node_size=node_size,
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

# plt.title("Steel Defect Detection - Logical Knowledge Graph", fontsize=20)
plt.axis("off")

## Save the graph visualization to Media
output_path = os.path.join(MEDIA_DIR, "Steel_Defect_Detection_KG.png")
plt.savefig(output_path, bbox_inches="tight", dpi=150)
print(f"Saved: {output_path}")

plt.show()

print("\nKnowledge Graph Summary:")
print(f"  Nodes: {G.number_of_nodes()}")
print(f"  Edges: {G.number_of_edges()}")
print(f"  Layers: {sorted(set(d['layer'] for _, d in G.nodes(data=True)))}")
