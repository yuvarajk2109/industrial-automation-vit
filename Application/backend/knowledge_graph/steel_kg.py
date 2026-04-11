"""
Steel Defect Detection Knowledge Graph
    - 5-layer directed graph
        - Input Evidence
        - Spatial Attributes
        - Interpretation
        - Quality Assessment
        - Decision & Action
 
Refer to Code/KG/steel.py
"""

import networkx as nx

# Thresholds for defect classification
AREA_THRESHOLD_MINOR = 1.0      # T1: below this = isolated minor
AREA_THRESHOLD_SEVERE = 5.0     # T2: above this = localised severe
DENSITY_THRESHOLD = 3           # T4: number of detected classes to be "widespread"


def build_steel_kg() -> nx.DiGraph:
    """
    Build the Steel Defect Detection Logical Knowledge Graph

    Layers:
        1. Input Evidence        - Defect classes detected
        2. Spatial Attributes    - Area, length, density, distribution
        3. Interpretation        - Isolated, localised, widespread, critical
        4. Quality Assessment    - Acceptable, marginal, unacceptable
        5. Decision & Action     - Accept, downgrade, reject, manual inspection
    """
    G = nx.DiGraph()

    # Layer 1: Input Evidence
    layer_input = [
        "Defect_Class_1_Detected",
        "Defect_Class_2_Detected",
        "Defect_Class_3_Detected",
        "Defect_Class_4_Detected"
    ]

    # Layer 2: Spatial / Severity Attributes
    layer_attributes = [
        "Defect_Area",
        "Defect_Length",
        "Defect_Density",
        "Defect_Distribution"
    ]

    # Layer 3: Defect Interpretation
    layer_interpretation = [
        "Isolated_Minor_Defect",
        "Localized_Severe_Defect",
        "Widespread_Defect_Pattern",
        "Critical_Structural_Defect"
    ]

    # Layer 4: Quality Assessment
    layer_quality = [
        "Acceptable_Quality",
        "Marginal_Quality",
        "Unacceptable_Quality"
    ]

    # Layer 5: Decision & Action
    layer_decision = [
        "Accept_Strip",
        "Downgrade_Strip",
        "Reject_Strip",
        "Manual_Inspection_Required"
    ]

    # Add nodes with layer metadata
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

    # Edges: Evidence --> Attributes
    edges_evidence_to_attr = [
        ("Defect_Class_1_Detected", "Defect_Area", "if present"),
        ("Defect_Class_2_Detected", "Defect_Area", "if present"),
        ("Defect_Class_3_Detected", "Defect_Density", "if clustered"),
        ("Defect_Class_4_Detected", "Defect_Length", "if elongated"),
        ("Defect_Class_1_Detected", "Defect_Distribution", ""),
        ("Defect_Class_2_Detected", "Defect_Distribution", ""),
        ("Defect_Class_3_Detected", "Defect_Distribution", ""),
        ("Defect_Class_4_Detected", "Defect_Distribution", ""),
    ]

    # Edges: Attributes --> Interpretation
    edges_attr_to_interp = [
        ("Defect_Area", "Isolated_Minor_Defect", "area < T1"),
        ("Defect_Area", "Localized_Severe_Defect", "area >= T2"),
        ("Defect_Length", "Critical_Structural_Defect", "length >= T3"),
        ("Defect_Density", "Widespread_Defect_Pattern", "density >= T4"),
        ("Defect_Distribution", "Widespread_Defect_Pattern", "distributed"),
    ]

    # Edges: Interpretation --> Quality
    edges_interp_to_quality = [
        ("Isolated_Minor_Defect", "Acceptable_Quality", ""),
        ("Localized_Severe_Defect", "Marginal_Quality", ""),
        ("Widespread_Defect_Pattern", "Unacceptable_Quality", ""),
        ("Critical_Structural_Defect", "Unacceptable_Quality", ""),
    ]

    # Edges: Quality --> Decision
    edges_quality_to_decision = [
        ("Acceptable_Quality", "Accept_Strip", ""),
        ("Marginal_Quality", "Downgrade_Strip", ""),
        ("Unacceptable_Quality", "Reject_Strip", ""),
    ]

    # Add all edges
    for src, dst, cond in edges_evidence_to_attr:
        G.add_edge(src, dst, condition=cond)
    for src, dst, cond in edges_attr_to_interp:
        G.add_edge(src, dst, condition=cond)
    for src, dst, cond in edges_interp_to_quality:
        G.add_edge(src, dst, condition=cond)
    for src, dst, cond in edges_quality_to_decision:
        G.add_edge(src, dst, condition=cond)

    # Special safety edge
    G.add_edge(
        "Critical_Structural_Defect",
        "Manual_Inspection_Required",
        condition="safety-critical"
    )

    return G

def evaluate_steel_kg(defect_summary: dict) -> dict:
    """
    Given a defect summary from steel inference
        - traverse the KG
        - determine the quality assessment and final decision

    Args:
        - defect_summary
            - dict with class_1..class_4, each having
                - "detected": bool
                - "area_pct": float

    Returns:
        - Structured KG evaluation result
    """
    G = build_steel_kg()

    activated_nodes = []
    traversal_path = []

    # Layer 1: Activate detected defect classes
    detected_classes = []
    total_defect_area = 0.0

    for key, info in defect_summary.items():
        if info["detected"]:
            class_num = key.split("_")[1]  # "class_1" → "1"
            node_name = f"Defect_Class_{class_num}_Detected"
            activated_nodes.append(node_name)
            detected_classes.append(class_num)
            total_defect_area += info["area_pct"]

    # No defects detected
    if not detected_classes:
        return {
            "activated_nodes": [],
            "traversal_path": [],
            "defect_interpretation": "No_Defects_Detected",
            "quality_assessment": "Acceptable_Quality",
            "decision": "Accept_Strip",
            "requires_manual_inspection": False,
            "total_defect_area_pct": 0.0,
            "details": "No defects were detected in the steel strip."
        }

    # Layer 2: Determine spatial attributes
    activated_nodes.append("Defect_Area")
    activated_nodes.append("Defect_Distribution")

    if len(detected_classes) >= DENSITY_THRESHOLD:
        activated_nodes.append("Defect_Density")

    # Check for elongated defects
    if "4" in detected_classes:
        activated_nodes.append("Defect_Length")

    # Layer 3: Determine interpretation
    defect_interpretation = "Isolated_Minor_Defect"  # Default

    if total_defect_area >= AREA_THRESHOLD_SEVERE:
        defect_interpretation = "Localized_Severe_Defect"
        activated_nodes.append("Localized_Severe_Defect")
    elif total_defect_area < AREA_THRESHOLD_MINOR:
        defect_interpretation = "Isolated_Minor_Defect"
        activated_nodes.append("Isolated_Minor_Defect")

    if len(detected_classes) >= DENSITY_THRESHOLD:
        defect_interpretation = "Widespread_Defect_Pattern"
        activated_nodes.append("Widespread_Defect_Pattern")

    if "Defect_Length" in activated_nodes and total_defect_area >= AREA_THRESHOLD_SEVERE:
        defect_interpretation = "Critical_Structural_Defect"
        activated_nodes.append("Critical_Structural_Defect")

    # Layer 4: Quality assessment
    quality_map = {
        "Isolated_Minor_Defect": "Acceptable_Quality",
        "Localized_Severe_Defect": "Marginal_Quality",
        "Widespread_Defect_Pattern": "Unacceptable_Quality",
        "Critical_Structural_Defect": "Unacceptable_Quality",
    }
    quality_assessment = quality_map.get(defect_interpretation, "Marginal_Quality")
    activated_nodes.append(quality_assessment)

    # Layer 5: Decision
    decision_map = {
        "Acceptable_Quality": "Accept_Strip",
        "Marginal_Quality": "Downgrade_Strip",
        "Unacceptable_Quality": "Reject_Strip",
    }

    decision = decision_map.get(quality_assessment, "Manual_Inspection_Required")
    activated_nodes.append(decision)

    requires_manual = defect_interpretation == "Critical_Structural_Defect"
    if requires_manual:
        activated_nodes.append("Manual_Inspection_Required")

    # Build traversal path
    for node in activated_nodes:
        for successor in G.successors(node):
            if successor in activated_nodes:
                edge_data = G.edges[node, successor]
                traversal_path.append({
                    "from": node,
                    "to": successor,
                    "condition": edge_data.get("condition", "")
                })

    # Build details string
    details_parts = [
        f"Detected defect classes: {', '.join(detected_classes)}",
        f"Total defect area: {total_defect_area:.2f}%",
        f"Interpretation: {defect_interpretation.replace('_', ' ')}",
        f"Quality: {quality_assessment.replace('_', ' ')}",
        f"Decision: {decision.replace('_', ' ')}"
    ]
    if requires_manual:
        details_parts.append("[WARNING] Manual inspection required - safety-critical defect detected")

    return {
        "activated_nodes": list(set(activated_nodes)),
        "traversal_path": traversal_path,
        "defect_interpretation": defect_interpretation,
        "quality_assessment": quality_assessment,
        "decision": decision,
        "requires_manual_inspection": requires_manual,
        "total_defect_area_pct": round(total_defect_area, 4),
        "details": " | ".join(details_parts)
    }