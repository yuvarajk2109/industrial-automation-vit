"""
Sugar Crystallisation Knowledge Graph
    - State
    - Condition
    - Action

Refer to Code/KG/sugar.py
"""

import networkx as nx

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


def infer_supersaturation_range(crystal_state: str, confidence: float) -> tuple:
    low, high = STATE_TEMPLATES[crystal_state]["supersaturation_ratio"]
    center = (low + high) / 2
    width = (high - low) * (1 - confidence)
    return (round(center - width / 2, 4), round(center + width / 2, 4))


def build_inferred_state(vit_output: dict) -> dict:
    """
    Build inferred process state from ViT classification output

    Args:
        - vit_output
            - "crystal_state": str
            - "confidence": float

    Returns:
        - Enriched state dict with 
            - nucleation risk
            - growth stability
    """
    cs = vit_output["crystal_state"]
    conf = vit_output["confidence"]

    return {
        "crystal_state": cs,
        "confidence": conf,
        "supersaturation_ratio_range": infer_supersaturation_range(cs, conf),
        "nucleation_risk": STATE_TEMPLATES[cs]["nucleation_risk"],
        "growth_stability": STATE_TEMPLATES[cs]["growth_stability"]
    }


def build_sugar_kg() -> nx.DiGraph:
    """
    Build the Unified Sugar Crystallisation Knowledge Graph

    Structure:
        - States        (observed)
        - Conditions    (derived)
        - Actions       (control)
    """
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

    KG.add_edge("UNSATURATED", "METASTABLE", edge_type="state_transition", trigger="increase_supersaturation")
    KG.add_edge("METASTABLE", "INTERMEDIATE", edge_type="state_transition", trigger="excess_evaporation")
    KG.add_edge("INTERMEDIATE", "LABILE", edge_type="state_transition", trigger="runaway_supersaturation")
    KG.add_edge("LABILE", "INTERMEDIATE", edge_type="state_transition", trigger="cooling_or_dilution")
    KG.add_edge("INTERMEDIATE", "METASTABLE", edge_type="state_transition", trigger="growth_stabilization")

    KG.add_edge("UNSATURATED", "low_nucleation_risk", edge_type="implies_condition")
    KG.add_edge("METASTABLE", "low_nucleation_risk", edge_type="implies_condition")
    KG.add_edge("METASTABLE", "stable_growth", edge_type="implies_condition")
    KG.add_edge("INTERMEDIATE", "medium_nucleation_risk", edge_type="implies_condition")
    KG.add_edge("INTERMEDIATE", "unstable_growth", edge_type="implies_condition")
    KG.add_edge("LABILE", "high_nucleation_risk", edge_type="implies_condition")
    KG.add_edge("LABILE", "unstable_growth", edge_type="implies_condition")

    KG.add_edge("low_nucleation_risk", "hold_process", edge_type="triggers_action")
    KG.add_edge("stable_growth", "hold_process", edge_type="triggers_action")
    KG.add_edge("unstable_growth", "increase_evaporation", edge_type="triggers_action")
    KG.add_edge("high_nucleation_risk", "reduce_seeding", edge_type="triggers_action")
    KG.add_edge("high_nucleation_risk", "emergency_dilution", edge_type="triggers_action")

    return KG


def _infer_actions(graph: nx.DiGraph, active_state: str) -> list:
    """
    Traverse KG from state node to find all triggered actions.
    """
    actions = set()

    for _, cond in graph.out_edges(active_state):
        if graph.nodes[cond].get("node_type") == "condition":
            for _, act in graph.out_edges(cond):
                if graph.nodes[act].get("node_type") == "action":
                    actions.add(act)

    return sorted(list(actions))


def _score_transitions(graph: nx.DiGraph, active_state: str) -> dict:
    """
    Score possible state transitions based
        - risk
        - controllability
    """
    scores = {}
    risk_weight = {"low": 0.2, "medium": 0.5, "high": 0.8}
    ctrl_weight = {"high": 0.9, "medium": 0.6, "low": 0.3}

    for _, dst, data in graph.out_edges(active_state, data=True):
        if data.get("edge_type") == "state_transition":
            risk = data.get("risk", "medium")
            ctrl = data.get("controllability", "medium")
            scores[dst] = round(
                risk_weight.get(risk, 0.5) * (1 - ctrl_weight.get(ctrl, 0.5)), 4
            )

    return scores


def evaluate_sugar_kg(prediction: dict) -> dict:
    """
    Given a sugar classification prediction
        - traverse the KG
        - determine 
            - recommended actions
            - state info

    Args:
        - prediction: dict from sugar_inference with
            - predicted_class
            - confidence

    Returns:
        - Structured KG evaluation result
    """
    KG = build_sugar_kg()

    state_map = {
        "unsaturated": "UNSATURATED",
        "metastable": "METASTABLE",
        "intermediate": "INTERMEDIATE",
        "labile": "LABILE"
    }

    crystal_state = state_map.get(
        prediction["predicted_class"], "INTERMEDIATE"
    )
    confidence = prediction["confidence"]

    vit_output = {
        "crystal_state": prediction["predicted_class"],
        "confidence": confidence
    }
    inferred = build_inferred_state(vit_output)

    recommended_actions = _infer_actions(KG, crystal_state)
    state_transitions = _score_transitions(KG, crystal_state)

    traversal_path = []
    activated_nodes = [crystal_state]

    for _, cond in KG.out_edges(crystal_state):
        if KG.nodes[cond].get("node_type") == "condition":
            activated_nodes.append(cond)
            edge_data = KG.edges[crystal_state, cond]
            traversal_path.append({
                "from": crystal_state,
                "to": cond,
                "edge_type": edge_data.get("edge_type", "")
            })

            for _, act in KG.out_edges(cond):
                if KG.nodes[act].get("node_type") == "action":
                    activated_nodes.append(act)
                    edge_data_act = KG.edges[cond, act]
                    traversal_path.append({
                        "from": cond,
                        "to": act,
                        "edge_type": edge_data_act.get("edge_type", "")
                    })

    state_info = KG.nodes[crystal_state]

    action_str = ", ".join(a.replace("_", " ") for a in recommended_actions)
    details = (
        f"Crystal state: {crystal_state} ({state_info.get('description', '')}) | "
        f"Nucleation risk: {inferred['nucleation_risk']} | "
        f"Growth stability: {inferred['growth_stability']} | "
        f"Supersaturation range: {inferred['supersaturation_ratio_range']} | "
        f"Recommended actions: {action_str}"
    )

    return {
        "crystal_state": crystal_state,
        "supersaturation_range": inferred["supersaturation_ratio_range"],
        "nucleation_risk": inferred["nucleation_risk"],
        "growth_stability": inferred["growth_stability"],
        "recommended_actions": recommended_actions,
        "state_transitions": state_transitions,
        "activated_nodes": activated_nodes,
        "traversal_path": traversal_path,
        "details": details
    }