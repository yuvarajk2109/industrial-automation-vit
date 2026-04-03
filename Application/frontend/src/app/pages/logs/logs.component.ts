import { Component, OnInit } from '@angular/core';
import { ApiService } from '../../core/services/api.service';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';
import { DatePipe } from '@angular/common';

@Component({
  selector: 'app-logs',
  standalone: true,
  imports: [StatusBadgeComponent, DatePipe],
  templateUrl: './logs.component.html',
  styleUrl: './logs.component.css'
})
export class LogsComponent implements OnInit {
  logs: any[] = [];
  total = 0;
  page = 1;
  limit = 20;
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
}
