import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-status-badge',
  standalone: true,
  templateUrl: './status-badge.component.html',
  styleUrl: './status-badge.component.css'
})
export class StatusBadgeComponent {
  @Input() label = '';
  @Input() variant: 'success' | 'warning' | 'error' | 'info' | 'steel' | 'sugar' | 'neutral' = 'neutral';
  @Input() showDot = false;
}
