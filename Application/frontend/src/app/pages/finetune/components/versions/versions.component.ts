import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { LoadingSpinnerComponent } from '../../../../shared/components/loading-spinner/loading-spinner.component';
import { PaginationComponent } from '../../../../shared/components/pagination/pagination.component';
import { DataTableComponent, TableColumn } from '../../../../shared/components/data-table/data-table.component';

@Component({
  selector: 'app-versions',
  standalone: true,
  imports: [CommonModule, StatusBadgeComponent, LoadingSpinnerComponent, PaginationComponent, DataTableComponent],
  templateUrl: './versions.component.html',
  styleUrl: './versions.component.css'
})
export class VersionsComponent {
  @Input() modelVersions: any[] = [];
  @Input() versionsLoading = false;
  @Input() isRollingBack = false;

  @Output() rollback = new EventEmitter<{domain: string, version: number}>();

  versionColumns: TableColumn[] = [
    { label: 'Created', width: '180px' },
    { label: 'Version', width: '80px' },
    { label: 'Domain', width: '80px' },
    { label: 'Filename', width: '1fr' },
    { label: 'Accuracy', width: '100px' },
    { label: 'Status / Actions', width: '150px' }
  ];

  page = 1;
  limit = 10;

  get totalPages(): number {
    return Math.ceil(this.modelVersions.length / this.limit) || 1;
  }

  get paginatedVersions(): any[] {
    const start = (this.page - 1) * this.limit;
    return this.modelVersions.slice(start, start + this.limit);
  }

  onPageChange(p: number) { this.page = p; }
  onLimitChange(l: number) { this.limit = l; this.page = 1; }

  formatDate(dateStr: any): string {
    if (!dateStr) return '-';
    try {
      const d1 = new Date(dateStr);
      if (!isNaN(d1.getTime())) return d1.toLocaleString();
      if (typeof dateStr === 'string') {
        let dStr = dateStr;
        if (!dStr.includes('T')) dStr = dStr.replace(' ', 'T');
        if (!dStr.endsWith('Z') && !dStr.includes('+')) dStr += 'Z';
        const d2 = new Date(dStr);
        if (!isNaN(d2.getTime())) return d2.toLocaleString();
      }
      return String(dateStr);
    } catch {
      return String(dateStr);
    }
  }
}
