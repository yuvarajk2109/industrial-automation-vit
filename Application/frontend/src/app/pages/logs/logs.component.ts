import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../core/services/api.service';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { DatePipe } from '@angular/common';
import { MarkdownPipe } from '../../shared/pipes/markdown.pipe';

@Component({
  selector: 'app-logs',
  standalone: true,
  imports: [StatusBadgeComponent, DatePipe, MarkdownPipe],
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

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.loadLogs();
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

  selectLog(log: any): void {
    this.selectedLog = this.selectedLog === log ? null : log;
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
}
