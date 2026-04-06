import { Component, Input, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { StatusBadgeComponent } from '../../../../shared/components/status-badge/status-badge.component';
import { LoadingSpinnerComponent } from '../../../../shared/components/loading-spinner/loading-spinner.component';
import { PaginationComponent } from '../../../../shared/components/pagination/pagination.component';
import { DataTableComponent, TableColumn } from '../../../../shared/components/data-table/data-table.component';
import { DropdownComponent } from '../../../../shared/components/dropdown/dropdown';
import { FormatResultPipe } from '../../../../shared/pipes/format-result.pipe';

@Component({
  selector: 'app-corrections',
  standalone: true,
  imports: [CommonModule, FormsModule, StatusBadgeComponent, LoadingSpinnerComponent, PaginationComponent, DataTableComponent, DropdownComponent, FormatResultPipe],
  templateUrl: './corrections.component.html',
  styleUrl: './corrections.component.css'
})
export class CorrectionsComponent {
  @Input() corrections: any[] = [];
  @Input() correctionsLoading = false;
  @Input() correctionFilter: 'all' | 'sugar' | 'steel' = 'all';
  @Input() correctionStatusFilter: 'all' | 'pending' | 'used' = 'all';
  @Input() correctionPage = 1;
  @Input() correctionTotalPages = 1;
  @Input() correctionLimit = 15;

  @Output() correctionFilterChange = new EventEmitter<'all' | 'sugar' | 'steel'>();
  @Output() correctionStatusFilterChange = new EventEmitter<'all' | 'pending' | 'used'>();
  @Output() filterChanged = new EventEmitter<void>();
  @Output() pageChange = new EventEmitter<number>();
  @Output() limitChange = new EventEmitter<number>();

  correctionColumns: TableColumn[] = [
    { label: 'Submitted', width: '180px' },
    { label: 'Image', width: '1fr' },
    { label: 'Domain', width: '80px' },
    { label: 'Original Prediction', width: '150px' },
    { label: 'Corrected As', width: '150px' },
    { label: 'Status', width: '100px' }
  ];

  onFilterChangeLocal() {
    this.filterChanged.emit();
  }

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

  domainOptions = [
    { label: 'All Domains', value: 'all' },
    { label: 'Sugar', value: 'sugar' },
    { label: 'Steel', value: 'steel' }
  ];

  statusOptions = [
    { label: 'All Statuses', value: 'all' },
    { label: 'Pending', value: 'pending' },
    { label: 'Used', value: 'used' }
  ];

  onDomainFilterChange(option: any) {
    if (option) {
      this.correctionFilterChange.emit(option.value);
      this.onFilterChangeLocal();
    }
  }

  onStatusFilterChange(option: any) {
    if (option) {
      this.correctionStatusFilterChange.emit(option.value);
      this.onFilterChangeLocal();
    }
  }
}
