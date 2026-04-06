import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { SimulationService } from '../../core/services/simulation.service';
import { SimulationState, SimulationSummary, CompletedImage, SimulationEvent } from '../../core/models/simulation.model';
import { PipelineVisualiserComponent } from '../../shared/components/pipeline-visualiser/pipeline-visualiser.component';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { DropdownComponent } from '../../shared/components/dropdown/dropdown';
import { ApiService } from '../../core/services/api.service';
import { FormatResultPipe } from '../../shared/pipes/format-result.pipe';
import { ImageCardComponent } from '../../shared/components/image-card/image-card.component';

@Component({
  selector: 'app-simulation',
  standalone: true,
  imports: [FormsModule, RouterLink, PipelineVisualiserComponent, StatusBadgeComponent, DropdownComponent, FormatResultPipe, ImageCardComponent],
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
    if (!this.steelDir && !this.sugarDir) return;

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
          image_path: event.image_path || '',
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
    if (ms <= 0) return '-';
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

  // - Correction / Feedback -
  flaggedImage: CompletedImage | null = null;
  correctedSugarClass = '';
  steelCorrections: { original_class: string; corrected_class: string; action: string }[] = [];
  missedDefects: string[] = [];
  correctionReason = '';
  isSubmittingCorrection = false;
  correctionSubmittedIds = new Set<number>();
  totalCorrectionsSubmitted = 0;

  flagImage(img: CompletedImage): void {
    if (this.flaggedImage === img) {
      this.flaggedImage = null;
      return;
    }
    this.flaggedImage = img;
    this.correctionReason = '';
    this.missedDefects = [];

    // Dropdown prep
    if (img.domain === 'sugar') {
      this.correctedSugarClass = '';
      this.selectedSugarOption = null;
    } else {
      // Build steel corrections from prediction
      this.steelCorrections = [];
      const defectSummary = img.prediction?.defect_summary || {};
      for (const [key, val] of Object.entries(defectSummary)) {
        if ((val as any).detected) {
          const cls = key.replace('class_', '');
          this.steelCorrections.push({
            original_class: cls,
            corrected_class: cls,
            action: 'keep'
          });
        }
      }
    }
  }

  addMissedDefect(cls: string): void {
    if (cls && !this.missedDefects.includes(cls)) {
      this.missedDefects.push(cls);
    }
  }

  removeMissedDefect(cls: string): void {
    this.missedDefects = this.missedDefects.filter(d => d !== cls);
  }

  submitSimCorrection(): void {
    if (!this.flaggedImage || !this.flaggedImage.log_id) return;

    let correctedLabel: any;
    const img = this.flaggedImage;

    if (img.domain === 'sugar') {
      if (this.correctedSugarClass === img.prediction?.predicted_class) return;
      if (!this.correctedSugarClass) return; // Prevent empty submission
      correctedLabel = { class: this.correctedSugarClass };
    } else {
      const corrections = this.steelCorrections
        .filter(c => c.corrected_class !== c.original_class || c.action === 'remove')
        .map(c => ({
          original_class: c.original_class,
          corrected_class: c.action === 'remove' ? 'none' : c.corrected_class,
          action: c.action
        }));
      if (corrections.length === 0 && this.missedDefects.length === 0) return;
      correctedLabel = {
        type: 'region_override',
        corrections,
        missed_defects: this.missedDefects
      };
    }

    this.isSubmittingCorrection = true;
    this.api.submitFeedback({
      log_id: img.log_id,
      domain: img.domain,
      corrected_label: correctedLabel,
      reason: this.correctionReason
    }).subscribe({
      next: (res: any) => {
        this.correctionSubmittedIds.add(img.index);
        this.totalCorrectionsSubmitted = res.pending_count || (this.totalCorrectionsSubmitted + 1);
        this.flaggedImage = null;
        this.isSubmittingCorrection = false;
      },
      error: () => {
        this.isSubmittingCorrection = false;
      }
    });
  }

  Math = Math;

  // Dropdown options
  sugarOptions = [
    { label: 'Unsaturated', value: 'unsaturated' },
    { label: 'Metastable', value: 'metastable' },
    { label: 'Intermediate', value: 'intermediate' },
    { label: 'Labile', value: 'labile' }
  ];

  get filteredSugarOptions() {
    const predicted = this.flaggedImage?.prediction?.predicted_class;
    return this.sugarOptions.filter(opt => opt.value !== predicted);
  }

  steelActionOptions(originalClass: string) {
    return [
      { label: `Keep as Class ${originalClass}`, value: 'keep' },
      { label: 'Reclassify', value: 'reclassify' },
      { label: 'Remove (False Positive)', value: 'remove' }
    ];
  }

  steelClassOptions(originalClass: string) {
    return ['1', '2', '3', '4']
      .filter(c => c !== originalClass)
      .map(c => ({ label: `Class ${c}`, value: c }));
  }

  selectedSugarOption: any = null;

  updateSugarCorrection(option: any): void {
    if (option) {
      this.correctedSugarClass = option.value;
      this.selectedSugarOption = option;
    } else {
      this.correctedSugarClass = '';
      this.selectedSugarOption = null;
    }
  }

  updateSteelAction(corr: any, option: any): void {
    if (option) {
      corr.action = option.value;
      if (corr.action === 'remove') {
        corr.corrected_class = 'none';
      } else {
        corr.corrected_class = corr.original_class;
      }
    }
  }

  updateSteelClass(corr: any, option: any): void {
    if (option) {
      corr.corrected_class = option.value;
    }
  }
}
