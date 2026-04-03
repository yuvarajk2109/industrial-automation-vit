export interface SteelDefectSummary {
  [key: string]: {
    detected: boolean;
    area_pct: number;
  };
}

export interface SteelPrediction {
  image_path: string;
  image_filename: string;
  domain: 'steel';
  defect_summary: SteelDefectSummary;
  dominant_defect: string;
  total_defect_area_pct: number;
  mask_overlay_path: string;
  raw_mask_path: string;
  inference_time_ms: number;
}

export interface SugarPrediction {
  image_path: string;
  image_filename: string;
  domain: 'sugar';
  predicted_class: string;
  confidence: number;
  all_probabilities: {
    unsaturated: number;
    metastable: number;
    intermediate: number;
    labile: number;
  };
  inference_time_ms: number;
}

export type Prediction = SteelPrediction | SugarPrediction;

export interface SteelKGResult {
  activated_nodes: string[];
  traversal_path: TraversalEdge[];
  defect_interpretation: string;
  quality_assessment: string;
  decision: string;
  requires_manual_inspection: boolean;
  total_defect_area_pct: number;
  details: string;
}

export interface SugarKGResult {
  crystal_state: string;
  supersaturation_range: [number, number];
  nucleation_risk: string;
  growth_stability: string;
  recommended_actions: string[];
  state_transitions: { [key: string]: number };
  activated_nodes: string[];
  traversal_path: TraversalEdge[];
  details: string;
}

export type KGResult = SteelKGResult | SugarKGResult;

export interface TraversalEdge {
  from: string;
  to: string;
  condition?: string;
  edge_type?: string;
}

export interface StepTimes {
  inference_ms: number;
  kg_ms: number;
  gemini_ms: number;
  db_ms: number;
}

export interface PipelineResult {
  log_id: string;
  session_id: string;
  domain: string;
  prediction: Prediction;
  knowledge_graph: KGResult;
  gemini_response: string;
  step_times: StepTimes;
  total_processing_ms: number;
}

export interface HealthCheck {
  status: string;
  service: string;
  device: string;
  mongodb: boolean;
}
