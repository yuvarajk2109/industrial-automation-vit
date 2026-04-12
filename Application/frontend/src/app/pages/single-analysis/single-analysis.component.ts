import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';
import { ChatService } from '../../core/services/chat.service';
import { PipelineResult } from '../../core/models/prediction.model';
import { PipelineVisualiserComponent } from '../../shared/components/pipeline-visualiser/pipeline-visualiser.component';
import { ImageCardComponent } from '../../shared/components/image-card/image-card.component';
import { KgGraphComponent } from '../../shared/components/kg-graph/kg-graph.component';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { DropdownComponent } from '../../shared/components/dropdown/dropdown';
import { KeyValuePipe } from '@angular/common';
import { MarkdownPipe } from '../../shared/pipes/markdown.pipe';
import { FormatResultPipe } from '../../shared/pipes/format-result.pipe';

@Component({
  selector: 'app-single-analysis',
  standalone: true,
  imports: [
    FormsModule,
    PipelineVisualiserComponent,
    ImageCardComponent,
    KgGraphComponent,
    StatusBadgeComponent,
    DropdownComponent,
    KeyValuePipe,
    MarkdownPipe,
    FormatResultPipe
  ],
  templateUrl: './single-analysis.component.html',
  styleUrl: './single-analysis.component.css'
})
export class SingleAnalysisComponent {
  protected readonly Math = Math;
  imagePath = '';
  selectedFile: File | null = null;
  localPreviewUrl: string | null = null;
  isHovering = false;
  isChatExpanded = false;
  domain: 'steel' | 'sugar' = 'steel';

  isAnalysing = false;
  currentStep = -1;
  result: PipelineResult | null = null;
  error: string | null = null;

  // Chat
  chatMessage = '';

  pipelineSteps = [
    { label: 'Image Loaded' },
    { label: 'DDA-ViT Inference' },
    { label: 'Knowledge Graph' },
    { label: 'Gemini AI' }
  ];

  constructor(
    private api: ApiService,
    public chatService: ChatService
  ) { }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.isHovering = true;
  }

  onDragLeave(event: DragEvent): void {
    this.isHovering = false;
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.isHovering = false;
    if (event.dataTransfer?.files.length) {
      if (this.localPreviewUrl) URL.revokeObjectURL(this.localPreviewUrl);
      this.selectedFile = event.dataTransfer.files[0];
      this.imagePath = this.selectedFile.name;
      this.localPreviewUrl = URL.createObjectURL(this.selectedFile);
    }
  }

  onFileSelect(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files?.length) {
      if (this.localPreviewUrl) URL.revokeObjectURL(this.localPreviewUrl);
      this.selectedFile = input.files[0];
      this.imagePath = this.selectedFile.name;
      this.localPreviewUrl = URL.createObjectURL(this.selectedFile);
    }
  }

  analyse(): void {
    if (!this.imagePath.trim() && !this.selectedFile) return;

    this.isAnalysing = true;
    this.result = null;
    this.error = null;
    this.correctionSubmitted = false;
    this.showCorrectionPanel = false;
    this.correctedSugarClass = '';
    this.selectedSugarOption = null;
    this.correctionReason = '';
    this.chatService.clearChat();

    // Simulate pipeline steps visually
    this.currentStep = 0;

    setTimeout(() => {
      this.currentStep = 1;

      const payload = this.selectedFile ? this.selectedFile : this.imagePath;

      this.api.predict(payload, this.domain).subscribe({
        next: (result) => {
          this.currentStep = 2;
          setTimeout(() => {
            this.currentStep = 3;
            setTimeout(() => {
              this.currentStep = 4;
              this.result = result;
              this.isAnalysing = false;

              // Initialize chatbot
              this.chatService.initializeChat(
                result.log_id,
                result.gemini_response
              );
            }, 300);
          }, 300);
        },
        error: (err) => {
          this.error = err.error?.error || 'Analysis failed. Please check the image path and try again.';
          this.isAnalysing = false;
          this.currentStep = -1;
        }
      });
    }, 400);
  }

  sendChat(): void {
    if (this.chatMessage.trim()) {
      this.chatService.sendMessage(this.chatMessage);
      this.chatMessage = '';
    }
  }

  /** Helper to extract KG data safely */
  get kgActivatedNodes(): string[] {
    return (this.result?.knowledge_graph as any)?.activated_nodes || [];
  }

  get kgTraversalPath(): any[] {
    return (this.result?.knowledge_graph as any)?.traversal_path || [];
  }

  get steelPrediction(): any {
    return this.result?.domain === 'steel' ? this.result.prediction : null;
  }

  get sugarPrediction(): any {
    return this.result?.domain === 'sugar' ? this.result.prediction : null;
  }

  get kgDetails(): string {
    return (this.result?.knowledge_graph as any)?.details || '';
  }

  // - Correction / Feedback -
  showCorrectionPanel = false;
  correctionSubmitted = false;
  correctedSugarClass = '';
  steelCorrections: { original_class: string; corrected_class: string; action: string }[] = [];
  missedDefects: string[] = [];
  correctionReason = '';
  pendingCorrectionCount = 0;
  isSubmittingCorrection = false;

  sugarOptions = [
    { label: 'Unsaturated', value: 'unsaturated' },
    { label: 'Metastable', value: 'metastable' },
    { label: 'Intermediate', value: 'intermediate' },
    { label: 'Labile', value: 'labile' }
  ];

  get filteredSugarOptions() {
    const predicted = this.sugarPrediction?.predicted_class;
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

  initSteelCorrections(): void {
    if (!this.steelPrediction) return;
    this.steelCorrections = [];
    for (const [key, val] of Object.entries(this.steelPrediction.defect_summary)) {
      if ((val as any).detected) {
        const cls = key.replace('class_', '');
        this.steelCorrections.push({
          original_class: cls,
          corrected_class: cls,  // default to same (no change)
          action: 'keep'
        });
      }
    }
    this.missedDefects = [];
  }

  toggleCorrectionPanel(): void {
    this.showCorrectionPanel = !this.showCorrectionPanel;
    if (this.showCorrectionPanel) {
      if (this.result?.domain === 'sugar') {
        this.correctedSugarClass = '';
        this.selectedSugarOption = null;
      } else if (this.result?.domain === 'steel') {
        this.initSteelCorrections();
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

  submitCorrection(): void {
    if (!this.result) return;

    let correctedLabel: any;

    if (this.result.domain === 'sugar') {
      if (this.correctedSugarClass === this.sugarPrediction?.predicted_class) {
        return;
      }
      if (!this.correctedSugarClass) {
        return;
      }
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
      log_id: this.result.log_id,
      domain: this.result.domain,
      corrected_label: correctedLabel,
      reason: this.correctionReason
    }).subscribe({
      next: (res) => {
        this.correctionSubmitted = true;
        this.pendingCorrectionCount = res.pending_count;
        this.showCorrectionPanel = false;
        this.isSubmittingCorrection = false;
      },
      error: (err) => {
        this.error = err.error?.error || 'Failed to submit correction.';
        this.isSubmittingCorrection = false;
      }
    });
  }
}