import { Component, Input, Output, EventEmitter, ContentChild, TemplateRef } from '@angular/core';
import { CommonModule } from '@angular/common';

export interface TableColumn {
  label: string;
  width: string;
}

@Component({
  selector: 'app-data-table',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './data-table.component.html',
  styleUrl: './data-table.component.css'
})
export class DataTableComponent {
  @Input() columns: TableColumn[] = [];
  @Input() data: any[] = [];
  @Input() selectedItem: any = null;
  @Input() emptyMessage = 'No data found.';
  @Input() isRowClickable: boolean = true;
  @Input() height: string = '550px';
  @Input() trackByProperty: string = '_id';
  @Input() fontSize: string = '0.85rem';

  @Output() rowClick = new EventEmitter<any>();

  @ContentChild('rowTemplate') rowTemplate!: TemplateRef<any>;

  get gridColumns() {
    return this.columns.map(c => c.width).join(' ');
  }

  trackByFn(index: number, item: any) {
    if (!item) return index;
    if (this.trackByProperty && item[this.trackByProperty]) {
      const val = item[this.trackByProperty];
      return typeof val === 'object' && val.$oid ? val.$oid : val;
    }
    return index;
  }

  onRowClick(item: any) {
    if (this.isRowClickable) {
      this.rowClick.emit(item);
    }
  }
}