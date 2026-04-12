"""
Sugar Knowledge Graph Tests
    - Validates KG traversal logic for all crystallisation states
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge_graph.sugar_kg import (
    build_sugar_kg, evaluate_sugar_kg,
    build_inferred_state, STATE_TEMPLATES
)


class TestBuildSugarKG:
    """
    - Tests for sugar KG construction
    """

    def test_kg_has_state_nodes(self):
        KG = build_sugar_kg()
        for state in ["UNSATURATED", "METASTABLE", "INTERMEDIATE", "LABILE"]:
            assert state in KG.nodes
            assert KG.nodes[state]["node_type"] == "state"

    def test_kg_has_condition_nodes(self):
        KG = build_sugar_kg()
        conditions = [
            "low_nucleation_risk", "medium_nucleation_risk",
            "high_nucleation_risk", "stable_growth", "unstable_growth"
        ]
        for cond in conditions:
            assert cond in KG.nodes
            assert KG.nodes[cond]["node_type"] == "condition"

    def test_kg_has_action_nodes(self):
        KG = build_sugar_kg()
        actions = [
            "hold_process", "increase_evaporation",
            "reduce_seeding", "emergency_dilution"
        ]
        for act in actions:
            assert act in KG.nodes
            assert KG.nodes[act]["node_type"] == "action"

    def test_kg_has_state_transitions(self):
        KG = build_sugar_kg()
        assert KG.has_edge("UNSATURATED", "METASTABLE")
        assert KG.has_edge("METASTABLE", "INTERMEDIATE")
        assert KG.has_edge("INTERMEDIATE", "LABILE")
        assert KG.has_edge("LABILE", "INTERMEDIATE")  # reverse transition

    def test_labile_connects_to_high_risk_and_unstable(self):
        KG = build_sugar_kg()
        assert KG.has_edge("LABILE", "high_nucleation_risk")
        assert KG.has_edge("LABILE", "unstable_growth")

    def test_high_risk_triggers_emergency_actions(self):
        KG = build_sugar_kg()
        assert KG.has_edge("high_nucleation_risk", "reduce_seeding")
        assert KG.has_edge("high_nucleation_risk", "emergency_dilution")


class TestBuildInferredState:
    """
    - Tests for state inference from ViT output
    """

    def test_metastable_inferred_state(self):
        state = build_inferred_state({
            "crystal_state": "metastable",
            "confidence": 0.9
        })
        assert state["crystal_state"] == "metastable"
        assert state["nucleation_risk"] == "low"
        assert state["growth_stability"] == "stable"
        assert isinstance(state["supersaturation_ratio_range"], tuple)
        assert len(state["supersaturation_ratio_range"]) == 2

    def test_labile_inferred_state(self):
        state = build_inferred_state({
            "crystal_state": "labile",
            "confidence": 0.85
        })
        assert state["nucleation_risk"] == "high"
        assert state["growth_stability"] == "unstable"

    def test_unsaturated_has_no_nucleation_risk(self):
        state = build_inferred_state({
            "crystal_state": "unsaturated",
            "confidence": 0.95
        })
        assert state["nucleation_risk"] == "none"
        assert state["growth_stability"] == "none"

    def test_high_confidence_narrows_supersaturation_range(self):
        high_conf = build_inferred_state({
            "crystal_state": "metastable", "confidence": 0.99
        })
        low_conf = build_inferred_state({
            "crystal_state": "metastable", "confidence": 0.5
        })
        high_range = high_conf["supersaturation_ratio_range"]
        low_range = low_conf["supersaturation_ratio_range"]

        high_width = high_range[1] - high_range[0]
        low_width = low_range[1] - low_range[0]
        assert high_width < low_width, "Higher confidence should give narrower range"


class TestEvaluateSugarKG:
    """
    - Tests for full sugar KG evaluation
    """

    def test_metastable_recommends_hold(self, sugar_prediction_metastable):
        result = evaluate_sugar_kg(sugar_prediction_metastable)

        assert result["crystal_state"] == "METASTABLE"
        assert result["nucleation_risk"] == "low"
        assert result["growth_stability"] == "stable"
        assert "hold_process" in result["recommended_actions"]

    def test_labile_recommends_emergency_actions(self, sugar_prediction_labile):
        result = evaluate_sugar_kg(sugar_prediction_labile)

        assert result["crystal_state"] == "LABILE"
        assert result["nucleation_risk"] == "high"
        assert result["growth_stability"] == "unstable"
        assert "emergency_dilution" in result["recommended_actions"]
        assert "reduce_seeding" in result["recommended_actions"]

    def test_unsaturated_recommends_hold(self, sugar_prediction_unsaturated):
        result = evaluate_sugar_kg(sugar_prediction_unsaturated)

        assert result["crystal_state"] == "UNSATURATED"
        assert result["nucleation_risk"] == "none"
        assert "hold_process" in result["recommended_actions"]

    def test_intermediate_recommends_evaporation(self, sugar_prediction_intermediate):
        result = evaluate_sugar_kg(sugar_prediction_intermediate)

        assert result["crystal_state"] == "INTERMEDIATE"
        assert result["nucleation_risk"] == "medium"
        assert "increase_evaporation" in result["recommended_actions"]

    def test_result_has_required_keys(self, sugar_prediction_metastable):
        result = evaluate_sugar_kg(sugar_prediction_metastable)

        required_keys = [
            "crystal_state", "supersaturation_range", "nucleation_risk",
            "growth_stability", "recommended_actions", "state_transitions",
            "activated_nodes", "traversal_path", "details"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_supersaturation_range_is_valid(self, sugar_prediction_labile):
        result = evaluate_sugar_kg(sugar_prediction_labile)

        low, high = result["supersaturation_range"]
        assert isinstance(low, float)
        assert isinstance(high, float)
        assert low < high, "Low bound must be less than high bound"
        assert low > 0, "Supersaturation must be positive"

    def test_traversal_path_is_populated(self, sugar_prediction_labile):
        result = evaluate_sugar_kg(sugar_prediction_labile)

        assert len(result["traversal_path"]) > 0
        for edge in result["traversal_path"]:
            assert "from" in edge
            assert "to" in edge
            assert "edge_type" in edge

    def test_activated_nodes_includes_state(self, sugar_prediction_metastable):
        result = evaluate_sugar_kg(sugar_prediction_metastable)

        assert "METASTABLE" in result["activated_nodes"]

    def test_state_transitions_are_scored(self, sugar_prediction_intermediate):
        result = evaluate_sugar_kg(sugar_prediction_intermediate)

        # Intermediate should have transitions to LABILE and METASTABLE
        transitions = result["state_transitions"]
        assert isinstance(transitions, dict)

    def test_details_string_has_info(self, sugar_prediction_labile):
        result = evaluate_sugar_kg(sugar_prediction_labile)

        assert "LABILE" in result["details"]
        assert "Nucleation risk" in result["details"]
        assert "Recommended actions" in result["details"]


class TestStateTemplates:
    """
    - Tests for STATE_TEMPLATES integrity
    """

    def test_all_four_states_defined(self):
        expected = ["unsaturated", "metastable", "intermediate", "labile"]
        for state in expected:
            assert state in STATE_TEMPLATES

    def test_each_state_has_required_fields(self):
        for state, template in STATE_TEMPLATES.items():
            assert "supersaturation_ratio" in template
            assert "nucleation_risk" in template
            assert "growth_stability" in template

    def test_supersaturation_ranges_are_ordered(self):
        for state, template in STATE_TEMPLATES.items():
            low, high = template["supersaturation_ratio"]
            assert low < high, f"{state}: {low} should be < {high}"