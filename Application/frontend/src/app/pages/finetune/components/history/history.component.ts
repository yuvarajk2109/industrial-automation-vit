import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { PaginationComponent } from '../../../../shared/components/pagination/pagination.component';
import { DataTableComponent, TableColumn } from '../../../../shared/components/data-table/data-table.component';

@Component({
  selector: 'app-history',
  standalone: true,
  imports: [CommonModule, StatusBadgeComponent, PaginationComponent, DataTableComponent],
  templateUrl: './history.component.html',
  styleUrl: './history.component.css'
})
export class HistoryComponent {
  @Input() jobHistory: any[] = [];
  
  historyColumns: TableColumn[] = [
    { label: 'Date', width: '180px' },
    { label: 'Job ID', width: '1fR' },
    { label: 'Domain', width: '80px' },
    { label: 'Corrections', width: '100px' },
    { label: 'Accuracy', width: '100px' },
    { label: 'Status', width: '120px' }
  ];

  page = 1;
  limit = 10;

  get totalPages(): number {
    return Math.ceil(this.jobHistory.length / this.limit) || 1;
  }

  get paginatedHistory(): any[] {
    const start = (this.page - 1) * this.limit;
    return this.jobHistory.slice(start, start + this.limit);
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
