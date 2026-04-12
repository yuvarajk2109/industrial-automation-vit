import { Component, Input } from '@angular/core';

@Component({
  selector: 'app-loading-spinner',
  standalone: true,
  templateUrl: './loading-spinner.component.html',
  styleUrl: './loading-spinner.component.css'
})
export class LoadingSpinnerComponent {
  @Input() size = 32;
  @Input() message = '';
  @Input() inline = false;
}