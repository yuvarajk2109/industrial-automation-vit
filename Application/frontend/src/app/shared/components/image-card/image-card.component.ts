import { Component, Input } from '@angular/core';
import { ApiService } from '../../../core/services/api.service';
import { StatusBadgeComponent } from '../status-badge/status-badge.component';

@Component({
  selector: 'app-image-card',
  standalone: true,
  imports: [StatusBadgeComponent],
  templateUrl: './image-card.component.html',
  styleUrl: './image-card.component.css'
})
export class ImageCardComponent {
  @Input() imagePath = '';
  @Input() overlayFilename: string | null = null;
  @Input() domainLabel: string | null = null;
  @Input() caption: string | null = null;

  showOverlay = false;

  constructor(private api: ApiService) { }

  getSourceUrl(): string {
    if (this.imagePath.startsWith('blob:')) {
      return this.imagePath;
    }
    return this.api.getSourceImageUrl(this.imagePath);
  }

  getOverlayUrl(): string {
    return this.api.getOutputImageUrl(this.overlayFilename!);
  }

  onImageError(event: Event): void {
    const img = event.target as HTMLImageElement;
    img.style.opacity = '0.3';
  }
}