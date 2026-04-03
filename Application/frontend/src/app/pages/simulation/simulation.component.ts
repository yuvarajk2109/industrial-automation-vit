import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { SimulationService } from '../../core/services/simulation.service';
import { SimulationState, SimulationSummary, CompletedImage, SimulationEvent } from '../../core/models/simulation.model';
import { PipelineVisualiserComponent } from '../../shared/components/pipeline-visualiser/pipeline-visualiser.component';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { ApiService } from '../../core/services/api.service';

@Component({
  selector: 'app-simulation',
  standalone: true,
  imports: [FormsModule, PipelineVisualiserComponent, StatusBadgeComponent],
  templateUrl: './simulation.component.html',
  styleUrl: './simulation.component.css'
})
export class SimulationComponent {
  steelDir = '';
  sugarDir = '';
  limit = 50;
  limitOptions = [10, 25, 50, 100, 0]; // 0 = All

  state: SimulationState = this.getInitialState();
  pipelineStep = -1;
  currentPrediction: any = null;
  currentKgResult: any = null;

  pipelineSteps = [
    { label: 'Image Loaded' },
    { label: 'DDA-ViT Inference' },
    { label: 'Knowledge Graph' },
    { label: 'Gemini AI' },
    { label: 'Logged to DB' }
  ];

  constructor(private simService: SimulationService, private api: ApiService) {}

  browseFolder(type: 'steel' | 'sugar'): void {
    this.api.browse('directory').subscribe({
      next: (res) => {
        if (res.path) {
          if (type === 'steel') this.steelDir = res.path;
          else this.sugarDir = res.path;
        }
      },
      error: (err) => console.error('Failed to browse', err)
    });
  }

  private getInitialState(): SimulationState {
    return {
      isRunning: false,
      currentStep: '',
      currentImage: '',
      currentDomain: '',
      currentIndex: 0,
      total: 0,
      processed: 0,
      progress: 0,
      etaMs: 0,
      summary: {
        steel: { accept: 0, downgrade: 0, reject: 0, manual_inspection: 0 },
        sugar: { unsaturated: 0, metastable: 0, intermediate: 0, labile: 0 }
      },
      completedImages: [],
      sessionId: ''
    };
  }

  startSimulation(): void {
    if (!this.steelDir || !this.sugarDir) return;

    this.state = this.getInitialState();
    this.state.isRunning = true;

    this.simService.startSimulation({
      steel_dir: this.steelDir,
      sugar_dir: this.sugarDir,
      limit: this.limit
    }).subscribe({
      next: (event: SimulationEvent) => this.handleEvent(event),
      error: (err) => {
        this.state.isRunning = false;
        console.error('Simulation error:', err);
      },
      complete: () => {
        this.state.isRunning = false;
      }
    });
  }

  stopSimulation(): void {
    this.simService.stopSimulation();
    this.state.isRunning = false;
  }

  private handleEvent(event: SimulationEvent): void {
    this.state.currentStep = event.step;

    switch (event.step) {
      case 'simulation_start':
        this.state.total = event.total || 0;
        this.state.sessionId = event.session_id || '';
        break;

      case 'image_start':
        this.state.currentImage = event.image || '';
        this.state.currentDomain = event.domain || '';
        this.state.currentIndex = event.index || 0;
        this.pipelineStep = 0;
        this.currentPrediction = null;
        this.currentKgResult = null;
        break;

      case 'inference_start':
        this.pipelineStep = 1;
        break;

      case 'inference_complete':
        this.pipelineStep = 2;
        this.currentPrediction = event.prediction;
        break;

      case 'kg_complete':
        this.pipelineStep = 3;
        this.currentKgResult = event.kg_result;
        break;

      case 'gemini_complete':
        this.pipelineStep = 4;
        break;

      case 'logged':
        this.pipelineStep = 5;
        break;

      case 'image_complete':
        this.state.processed = event.processed || (this.state.processed + 1);
        this.state.progress = event.progress || 0;
        this.state.etaMs = event.eta_ms || 0;
        if (event.summary) this.state.summary = event.summary;

        this.state.completedImages.push({
          index: event.index || 0,
          filename: event.image || '',
          domain: event.domain || '',
          prediction: this.currentPrediction,
          kg_result: this.currentKgResult,
          log_id: event.log_id || '',
          total_ms: event.total_ms || 0
        });
        break;

      case 'image_error':
        this.state.processed++;
        break;

      case 'simulation_complete':
        this.state.isRunning = false;
        if (event.summary) this.state.summary = event.summary;
        break;
    }
  }

  formatEta(ms: number): string {
    if (ms <= 0) return '–';
    const secs = Math.round(ms / 1000);
    if (secs < 60) return `${secs}s`;
    const mins = Math.floor(secs / 60);
    return `${mins}m ${secs % 60}s`;
  }

  getLimitLabel(val: number): string {
    return val === 0 ? 'All' : val.toString();
  }

  getDecisionBadge(img: CompletedImage): { label: string; variant: 'success' | 'warning' | 'error' | 'info' | 'steel' | 'sugar' | 'neutral' } {
    if (img.domain === 'steel') {
      const decision = img.kg_result?.decision || 'Unknown';
      if (decision.includes('Accept')) return { label: 'Accept', variant: 'success' };
      if (decision.includes('Downgrade')) return { label: 'Downgrade', variant: 'warning' };
      if (decision.includes('Reject')) return { label: 'Reject', variant: 'error' };
      return { label: decision, variant: 'neutral' };
    } else {
      const cls = img.prediction?.predicted_class || 'unknown';
      return { label: cls, variant: 'sugar' };
    }
  }

  Math = Math;
}
