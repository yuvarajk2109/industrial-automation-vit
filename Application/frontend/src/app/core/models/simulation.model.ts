export interface SimulationConfig {
  steel_dir: string;
  sugar_dir: string;
  limit: number;
}

export interface SimulationEvent {
  step: string;
  index?: number;
  total?: number;
  image?: string;
  image_path?: string;
  domain?: string;
  prediction?: any;
  kg_result?: any;
  response?: string;
  log_id?: string;
  total_ms?: number;
  progress?: number;
  processed?: number;
  eta_ms?: number;
  summary?: SimulationSummary;
  session_id?: string;
  total_steel?: number;
  total_sugar?: number;
  total_processed?: number;
  total_time_ms?: number;
  error?: string;
  time_ms?: number;
  limit_per_domain?: number;
}

export interface SimulationSummary {
  steel: {
    accept: number;
    downgrade: number;
    reject: number;
    manual_inspection: number;
  };
  sugar: {
    unsaturated: number;
    metastable: number;
    intermediate: number;
    labile: number;
  };
}

export interface CompletedImage {
  index: number;
  filename: string;
  image_path: string;
  domain: string;
  prediction: any;
  kg_result: any;
  log_id: string;
  total_ms: number;
}

export interface SimulationState {
  isRunning: boolean;
  currentStep: string;
  currentImage: string;
  currentDomain: string;
  currentIndex: number;
  total: number;
  processed: number;
  progress: number;
  etaMs: number;
  summary: SimulationSummary;
  completedImages: CompletedImage[];
  sessionId: string;
}
