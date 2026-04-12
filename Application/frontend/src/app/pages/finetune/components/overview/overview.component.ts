import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { LoadingSpinnerComponent } from '../../../../shared/components/loading-spinner/loading-spinner.component';

@Component({
  selector: 'app-overview',
  standalone: true,
  imports: [CommonModule, FormsModule, LoadingSpinnerComponent],
  templateUrl: './overview.component.html',
  styleUrl: './overview.component.css'
})
export class OverviewComponent {
  @Input() feedbackStats: any;
  @Input() feedbackLoading = true;
  @Input() activeJob: any;
  @Input() selectedDomain: 'sugar' | 'steel' = 'sugar';
  @Input() showAdvancedConfig = false;
  @Input() finetuneConfig: any;
  @Input() isStartingJob = false;
  @Input() canStartFineTune = false;

  @Output() selectedDomainChange = new EventEmitter<'sugar' | 'steel'>();
  @Output() showAdvancedConfigChange = new EventEmitter<boolean>();
  @Output() startFineTune = new EventEmitter<void>();

  getPendingCount(domain: string): number {
    return this.feedbackStats?.per_domain?.[domain]?.pending || 0;
  }

  getJobProgressPercent(): number {
    if (!this.activeJob?.progress) return 0;
    const { epoch, total_epochs } = this.activeJob.progress;
    return Math.round((epoch / total_epochs) * 100);
  }
}