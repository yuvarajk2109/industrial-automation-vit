import { Component, Input, Output, EventEmitter, OnChanges } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-pagination',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './pagination.component.html',
  styleUrl: './pagination.component.css'
})
export class PaginationComponent implements OnChanges {
  @Input() page: number = 1;
  @Input() totalPages: number = 1;
  @Input() limit: number = 20;
  @Input() limitOptions: number[] = [20, 30, 50];

  @Output() pageChange = new EventEmitter<number>();
  @Output() limitChange = new EventEmitter<number>();

  jumpPage: number = 1;

  ngOnChanges() {
    this.jumpPage = this.page;
  }

  setLimit(newLimit: number) {
    if (this.limit !== newLimit) {
      this.limitChange.emit(newLimit);
    }
  }

  goToPage(p: number) {
    if (p >= 1 && p <= this.totalPages && p !== this.page) {
      this.pageChange.emit(p);
    }
  }

  goToJumpPage() {
    if (this.jumpPage && this.jumpPage >= 1 && this.jumpPage <= this.totalPages) {
      this.goToPage(this.jumpPage);
    } else {
      this.jumpPage = this.page;
    }
  }
}