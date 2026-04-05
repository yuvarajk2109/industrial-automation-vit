import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../core/services/api.service';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { DatePipe } from '@angular/common';
import { MarkdownPipe } from '../../shared/pipes/markdown.pipe';
import { FormsModule } from '@angular/forms';
import { PaginationComponent } from '../../shared/components/pagination/pagination.component';
import { DataTableComponent, TableColumn } from '../../shared/components/data-table/data-table.component';
import { DropdownComponent } from '../../shared/components/dropdown/dropdown';

@Component({
  selector: 'app-logs',
  standalone: true,
  imports: [StatusBadgeComponent, DatePipe, MarkdownPipe, FormsModule, PaginationComponent, DataTableComponent, DropdownComponent],
  templateUrl: './logs.component.html',
  styleUrl: './logs.component.css'
})
export class LogsComponent implements OnInit {
  logs: any[] = [];
  total = 0;
  page = 1;
  limit = 20;
  limitOptions = [20, 30, 50];
  totalPages = 1;
  domainFilter = '';
  selectedLog: any = null;
  jumpPage: number = 1;

  tableColumns: TableColumn[] = [
    { label: 'Timestamp', width: '180px' },
    { label: 'Image', width: '1fr' },
    { label: 'Domain', width: '80px' },
    { label: 'Result', width: '150px' },
    { label: '', width: '30px' }, 
    { label: 'Time', width: '70px' }
  ];

  // - Correction State -
  correctionMode = false;
  correctedSugarClass = '';
  steelCorrections: { original_class: string; corrected_class: string; action: string }[] = [];
  missedDefects: string[] = [];
  correctionReason = '';
  isSubmittingCorrection = false;
  correctionSubmitted = false;
  pendingCorrectionCount = 0;
  correctedLogIds = new Set<string>();
  feedbackStats: any = null;

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.loadLogs();
    this.loadFeedbackStats();
  }

  loadFeedbackStats(): void {
    this.api.getFeedbackStats().subscribe({
      next: (res) => {
        this.feedbackStats = res;
      },
      error: () => {}
    });
  }

  loadLogs(): void {
    this.api.getLogs({
      domain: this.domainFilter || undefined,
      page: this.page,
      limit: this.limit
    }).subscribe({
      next: (data) => {
        this.logs = data.logs || [];
        this.total = data.total || 0;
        this.totalPages = data.total_pages || 1;
        this.jumpPage = this.page;
      },
      error: () => {}
    });
  }

  setFilter(domain: string): void {
    this.domainFilter = domain;
    this.page = 1;
    this.loadLogs();
  }

  goToPage(p: number): void {
    if (p < 1 || p > this.totalPages) return;
    this.page = p;
    this.loadLogs();
  }

  goToJumpPage(): void {
    if (this.jumpPage && this.jumpPage >= 1 && this.jumpPage <= this.totalPages) {
      this.goToPage(this.jumpPage);
    } else {
      this.jumpPage = this.page;
    }
  }

  selectLog(log: any): void {
    this.selectedLog = this.selectedLog === log ? null : log;
    this.correctionMode = false;
    this.correctionSubmitted = this.selectedLog ? this.isLogCorrected(this.selectedLog) : false;
  }

  setLimit(newLimit: number): void {
    if (this.limit === newLimit) return;
    this.limit = newLimit;
    this.page = 1;
    this.loadLogs();
  }

  getStartIndex(): number {
    if (this.total === 0) return 0;
    return (this.page - 1) * this.limit + 1;
  }

  getEndIndex(): number {
    return Math.min(this.page * this.limit, this.total);
  }

  getLogId(log: any): string {
    if (!log?._id) return '';
    return typeof log._id === 'string' ? log._id : log._id.$oid || '';
  }

  isLogCorrected(log: any): boolean {
    return this.correctedLogIds.has(this.getLogId(log)) || !!log.has_pending_correction;
  }

  // - Correction Methods -

  toggleCorrection(): void {
    this.correctionMode = !this.correctionMode;
    if (this.correctionMode && this.selectedLog) {
      this.correctionReason = '';
      this.missedDefects = [];
      if (this.selectedLog.domain === 'sugar') {
        this.correctedSugarClass = '';
        this.selectedSugarOption = null;
      } else {
        this.initSteelCorrections();
      }
    }
  }

  initSteelCorrections(): void {
    this.steelCorrections = [];
    const summary = this.selectedLog?.model_prediction?.defect_summary || {};
    for (const [key, val] of Object.entries(summary)) {
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

  addMissedDefect(cls: string): void {
    if (cls && !this.missedDefects.includes(cls)) {
      this.missedDefects.push(cls);
    }
  }

  removeMissedDefect(cls: string): void {
    this.missedDefects = this.missedDefects.filter(d => d !== cls);
  }

  submitLogCorrection(): void {
    if (!this.selectedLog) return;
    const logId = this.getLogId(this.selectedLog);
    if (!logId) return;

    let correctedLabel: any;

    if (this.selectedLog.domain === 'sugar') {
      if (this.correctedSugarClass === this.selectedLog.model_prediction?.predicted_class) return;
      if (!this.correctedSugarClass) return; // Prevent empty submission
      correctedLabel = { class: this.correctedSugarClass };
    } else {
      const corrections = this.steelCorrections
        .filter(c => c.action !== 'keep')
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
      log_id: logId,
      domain: this.selectedLog.domain,
      corrected_label: correctedLabel,
      reason: this.correctionReason
    }).subscribe({
      next: (res) => {
        this.correctionSubmitted = true;
        this.pendingCorrectionCount = res.pending_count;
        this.correctionMode = false;
        this.isSubmittingCorrection = false;
        this.correctedLogIds.add(logId);
      },
      error: () => {
        this.isSubmittingCorrection = false;
      }
    });
  }
  // Dropdown options
  sugarOptions = [
    { label: 'Unsaturated', value: 'unsaturated' },
    { label: 'Metastable', value: 'metastable' },
    { label: 'Intermediate', value: 'intermediate' },
    { label: 'Labile', value: 'labile' }
  ];

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

  getPendingCount(domain: string): number {
    return this.feedbackStats?.per_domain?.[domain]?.pending || 0;
  }
}

