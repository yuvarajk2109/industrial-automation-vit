import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule, DatePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ApiService } from '../../core/services/api.service';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { LoadingSpinnerComponent } from '../../shared/components/loading-spinner/loading-spinner.component';
import { PaginationComponent } from '../../shared/components/pagination/pagination.component';
import { DataTableComponent, TableColumn } from '../../shared/components/data-table/data-table.component';
import { OverviewComponent } from './components/overview/overview.component';
import { CorrectionsComponent } from './components/corrections/corrections.component';
import { VersionsComponent } from './components/versions/versions.component';
import { HistoryComponent } from './components/history/history.component';

@Component({
  selector: 'app-finetune',
  standalone: true,
  imports: [CommonModule, FormsModule, OverviewComponent, CorrectionsComponent, VersionsComponent, HistoryComponent],
  providers: [DatePipe],
  templateUrl: './finetune.component.html',
  styleUrl: './finetune.component.css'
})
export class FinetuneComponent implements OnInit, OnDestroy {
  feedbackStats: any = null;
  feedbackLoading = true;

  activeJob: any = null;
  jobPollingInterval: any;

  jobHistory: any[] = [];
  historyLoading = true;

  modelVersions: any[] = [];
  activeVersions: any = { sugar: 0, steel: 0 };
  versionsLoading = true;

  corrections: any[] = [];
  correctionsLoading = false;
  correctionFilter: 'all' | 'sugar' | 'steel' = 'all';
  correctionStatusFilter: 'all' | 'pending' | 'used' = 'all';
  correctionPage = 1;
  correctionTotal = 0;
  correctionTotalPages = 1;
  correctionLimit = 20;

  selectedDomain: 'sugar' | 'steel' = 'sugar';
  showAdvancedConfig = false;
  finetuneConfig = {
    lr: 0.0001,
    epochs: 10,
    min_corrections: 5,
    validation_split: 0.2,
    unfreeze_projection: false,
    early_stopping_patience: 3
  };

  error: string | null = null;
  successMessage: string | null = null;
  isStartingJob = false;
  isRollingBack = false;
  activeTab: 'overview' | 'corrections' | 'versions' | 'history' = 'overview';

  correctionColumns: TableColumn[] = [
    { label: 'Submitted', width: '180px' },
    { label: 'Image', width: '1fr' },
    { label: 'Domain', width: '80px' },
    { label: 'Original Prediction', width: '150px' },
    { label: 'Corrected As', width: '150px' },
    { label: 'Status', width: '100px' }
  ];

  versionColumns: TableColumn[] = [
    { label: 'Created', width: '180px' },
    { label: 'Version', width: '80px' },
    { label: 'Domain', width: '80px' },
    { label: 'Filename', width: '1fr' },
    { label: 'Accuracy', width: '100px' },
    { label: 'Status / Actions', width: '150px' }
  ];

  historyColumns: TableColumn[] = [
    { label: 'Date', width: '180px' },
    { label: 'Job ID', width: '1fR' },
    { label: 'Domain', width: '80px' },
    { label: 'Corrections', width: '100px' },
    { label: 'Accuracy', width: '100px' },
    { label: 'Status', width: '120px' }
  ];

  constructor(private api: ApiService, private datePipe: DatePipe) { }

  ngOnInit(): void {
    this.loadFeedbackStats();
    this.loadJobStatus();
    this.loadJobHistory();
    this.loadModelVersions();

    this.jobPollingInterval = setInterval(() => {
      if (this.activeJob?.status === 'running') {
        this.loadJobStatus();
      }
    }, 3000);
  }

  ngOnDestroy(): void {
    if (this.jobPollingInterval) {
      clearInterval(this.jobPollingInterval);
    }
  }

  loadFeedbackStats(): void {
    this.feedbackLoading = true;
    this.api.getFeedbackStats().subscribe({
      next: (stats) => {
        this.feedbackStats = stats;
        this.feedbackLoading = false;
      },
      error: () => {
        this.feedbackLoading = false;
      }
    });
  }

  loadJobStatus(): void {
    this.api.getFineTuneStatus().subscribe({
      next: (job) => {
        this.activeJob = job.status !== 'idle' ? job : null;
        if (job.status === 'completed' || job.status === 'failed') {
          this.loadFeedbackStats();
          this.loadJobHistory();
          this.loadModelVersions();
        }
      }
    });
  }

  loadJobHistory(): void {
    this.historyLoading = true;
    this.api.getFineTuneHistory().subscribe({
      next: (res) => {
        this.jobHistory = res.jobs || [];
        this.historyLoading = false;
      },
      error: () => {
        this.historyLoading = false;
      }
    });
  }

  loadModelVersions(): void {
    this.versionsLoading = true;
    this.api.getModelVersions().subscribe({
      next: (res) => {
        this.modelVersions = res.versions || [];
        this.activeVersions = res.active_versions || { sugar: 0, steel: 0 };
        this.versionsLoading = false;
      },
      error: () => {
        this.versionsLoading = false;
      }
    });
  }

  loadCorrections(): void {
    this.correctionsLoading = true;
    const params: any = { page: this.correctionPage, limit: this.correctionLimit };
    if (this.correctionFilter !== 'all') params.domain = this.correctionFilter;
    if (this.correctionStatusFilter !== 'all') params.status = this.correctionStatusFilter;

    this.api.getFeedback(params).subscribe({
      next: (res) => {
        this.corrections = res.feedback || [];
        this.correctionTotal = res.total || 0;
        this.correctionTotalPages = res.total_pages || 1;
        this.correctionsLoading = false;
      },
      error: () => {
        this.correctionsLoading = false;
      }
    });
  }

  startFineTune(): void {
    this.error = null;
    this.successMessage = null;
    this.isStartingJob = true;

    this.api.startFineTune(this.selectedDomain, this.finetuneConfig).subscribe({
      next: (res) => {
        this.isStartingJob = false;
        if (res.error) {
          this.error = res.error;
        } else {
          this.successMessage = res.message;
          this.activeJob = {
            job_id: res.job_id,
            status: 'running',
            domain: this.selectedDomain
          };
          this.loadJobStatus();
        }
      },
      error: (err) => {
        this.isStartingJob = false;
        this.error = err.error?.error || 'Failed to start fine-tune job.';
      }
    });
  }

  rollbackToVersion(domain: string, version: number): void {
    if (!confirm(`Rollback ${domain} model to version ${version}? The current model will be archived.`)) {
      return;
    }

    this.isRollingBack = true;
    this.error = null;

    this.api.rollbackModel(domain, version).subscribe({
      next: (res) => {
        this.isRollingBack = false;
        this.successMessage = `Rolled back ${domain} model to version ${version}`;
        this.loadModelVersions();
      },
      error: (err) => {
        this.isRollingBack = false;
        this.error = err.error?.error || 'Rollback failed.';
      }
    });
  }

  switchTab(tab: 'overview' | 'corrections' | 'versions' | 'history'): void {
    this.activeTab = tab;
    if (tab === 'corrections' && this.corrections.length === 0) {
      this.loadCorrections();
    }
  }

  onCorrectionFilterChange(): void {
    this.correctionPage = 1;
    this.loadCorrections();
  }

  setCorrectionLimit(limit: number): void {
    this.correctionLimit = limit;
    this.correctionPage = 1;
    this.loadCorrections();
  }

  goToCorrectionPage(page: number): void {
    if (page >= 1 && page <= this.correctionTotalPages) {
      this.correctionPage = page;
      this.loadCorrections();
    }
  }

  get canStartFineTune(): boolean {
    if (this.isStartingJob) return false;
    if (this.activeJob?.status === 'running') return false;
    if (!this.feedbackStats) return false;

    const pending = this.feedbackStats.per_domain?.[this.selectedDomain]?.pending || 0;
    return pending >= this.finetuneConfig.min_corrections;
  }

  getPendingCount(domain: string): number {
    return this.feedbackStats?.per_domain?.[domain]?.pending || 0;
  }

  getJobProgressPercent(): number {
    if (!this.activeJob?.progress) return 0;
    const { epoch, total_epochs } = this.activeJob.progress;
    return Math.round((epoch / total_epochs) * 100);
  }

  formatDate(dateStr: string): string {
    if (!dateStr) return '-';
    try {
      let dStr = dateStr;
      if (!dStr.endsWith('Z')) {
        dStr = dStr.replace(' ', 'T') + 'Z';
      }
      return new Date(dStr).toLocaleString();
    } catch {
      return dateStr;
    }
  }
}
