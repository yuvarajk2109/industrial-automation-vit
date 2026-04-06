import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';
import { ChatService } from '../../core/services/chat.service';
import { PipelineResult } from '../../core/models/prediction.model';
import { PipelineVisualiserComponent } from '../../shared/components/pipeline-visualiser/pipeline-visualiser.component';
import { ImageCardComponent } from '../../shared/components/image-card/image-card.component';
import { KgGraphComponent } from '../../shared/components/kg-graph/kg-graph.component';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { KeyValuePipe } from '@angular/common';
import { MarkdownPipe } from '../../shared/pipes/markdown.pipe';

@Component({
  selector: 'app-single-analysis',
  standalone: true,
  imports: [
    FormsModule,
    PipelineVisualiserComponent,
    ImageCardComponent,
    KgGraphComponent,
    StatusBadgeComponent,
    KeyValuePipe,
    MarkdownPipe
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
  ) {}

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
}
