"""
Steel Knowledge Graph Tests
    - Validates KG traversal logic for all defect scenarios
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from knowledge_graph.steel_kg import build_steel_kg, evaluate_steel_kg


class TestBuildSteelKG:
    """
    - Tests for KG construction
    """

    def test_kg_has_correct_node_count(self):
        G = build_steel_kg()
        # 4 input + 4 attributes + 4 interpretation + 3 quality + 4 decision = 19
        assert len(G.nodes) == 19

    def test_kg_layers_are_assigned(self):
        G = build_steel_kg()
        layer_counts = {}
        for _, data in G.nodes(data=True):
            layer = data.get("layer", 0)
            layer_counts[layer] = layer_counts.get(layer, 0) + 1

        assert layer_counts[1] == 4   # Input Evidence
        assert layer_counts[2] == 4   # Spatial Attributes
        assert layer_counts[3] == 4   # Interpretation
        assert layer_counts[4] == 3   # Quality Assessment
        assert layer_counts[5] == 4   # Decision

    def test_kg_has_safety_edge(self):
        G = build_steel_kg()
        assert G.has_edge("Critical_Structural_Defect", "Manual_Inspection_Required")
        edge_data = G.edges["Critical_Structural_Defect", "Manual_Inspection_Required"]
        assert edge_data["condition"] == "safety-critical"

    def test_kg_has_decision_edges(self):
        G = build_steel_kg()
        assert G.has_edge("Acceptable_Quality", "Accept_Strip")
        assert G.has_edge("Marginal_Quality", "Downgrade_Strip")
        assert G.has_edge("Unacceptable_Quality", "Reject_Strip")


class TestEvaluateSteelKG:
    """
    - Tests for 
        - KG evaluation
        - traversal logic
    """

    def test_no_defects_yields_accept(self, steel_prediction_no_defects):
        result = evaluate_steel_kg(steel_prediction_no_defects["defect_summary"])

        assert result["defect_interpretation"] == "No_Defects_Detected"
        assert result["quality_assessment"] == "Acceptable_Quality"
        assert result["decision"] == "Accept_Strip"
        assert result["requires_manual_inspection"] is False
        assert result["total_defect_area_pct"] == 0.0
        assert len(result["activated_nodes"]) == 0

    def test_minor_defect_yields_accept(self, steel_prediction_minor_defect):
        result = evaluate_steel_kg(steel_prediction_minor_defect["defect_summary"])

        assert result["defect_interpretation"] == "Isolated_Minor_Defect"
        assert result["quality_assessment"] == "Acceptable_Quality"
        assert result["decision"] == "Accept_Strip"
        assert result["requires_manual_inspection"] is False

    def test_severe_defect_yields_downgrade(self, steel_prediction_severe_defect):
        result = evaluate_steel_kg(steel_prediction_severe_defect["defect_summary"])

        assert result["defect_interpretation"] == "Localized_Severe_Defect"
        assert result["quality_assessment"] == "Marginal_Quality"
        assert result["decision"] == "Downgrade_Strip"
        assert result["requires_manual_inspection"] is False

    def test_critical_defect_yields_reject_and_manual(self, steel_prediction_critical_defect):
        result = evaluate_steel_kg(steel_prediction_critical_defect["defect_summary"])

        assert result["defect_interpretation"] == "Critical_Structural_Defect"
        assert result["quality_assessment"] == "Unacceptable_Quality"
        assert result["decision"] == "Reject_Strip"
        assert result["requires_manual_inspection"] is True
        assert "Manual_Inspection_Required" in result["activated_nodes"]

    def test_widespread_defect_yields_reject(self, steel_prediction_widespread):
        result = evaluate_steel_kg(steel_prediction_widespread["defect_summary"])

        assert result["defect_interpretation"] == "Widespread_Defect_Pattern"
        assert result["quality_assessment"] == "Unacceptable_Quality"
        assert result["decision"] == "Reject_Strip"

    def test_result_has_required_keys(self, steel_prediction_minor_defect):
        result = evaluate_steel_kg(steel_prediction_minor_defect["defect_summary"])

        required_keys = [
            "activated_nodes", "traversal_path", "defect_interpretation",
            "quality_assessment", "decision", "requires_manual_inspection",
            "total_defect_area_pct", "details"
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_traversal_path_has_valid_edges(self, steel_prediction_severe_defect):
        result = evaluate_steel_kg(steel_prediction_severe_defect["defect_summary"])

        for edge in result["traversal_path"]:
            assert "from" in edge
            assert "to" in edge
            assert "condition" in edge

    def test_details_contains_decision_info(self, steel_prediction_critical_defect):
        result = evaluate_steel_kg(steel_prediction_critical_defect["defect_summary"])

        assert "WARNING" in result["details"]
        assert "Manual inspection required" in result["details"]
        assert "safety-critical" in result["details"]

    def test_total_defect_area_is_sum(self, steel_prediction_severe_defect):
        summary = steel_prediction_severe_defect["defect_summary"]
        result = evaluate_steel_kg(summary)

        expected_total = sum(
            v["area_pct"] for v in summary.values() if v["detected"]
        )
        assert abs(result["total_defect_area_pct"] - expected_total) < 0.01