import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-pipeline-visualiser',
  standalone: true,
  templateUrl: './pipeline-visualiser.component.html',
  styleUrl: './pipeline-visualiser.component.css'
})
export class PipelineVisualiserComponent {
  @Input() currentStep = -1;
  @Input() orientation: 'horizontal' | 'vertical' = 'horizontal';
  @Input() steps: { label: string; preview?: string }[] = [
    { label: 'Image Loaded' },
    { label: 'DDA-ViT Inference' },
    { label: 'Knowledge Graph' },
    { label: 'Gemini AI' },
    { label: 'Logged to DB' }
  ];
}
