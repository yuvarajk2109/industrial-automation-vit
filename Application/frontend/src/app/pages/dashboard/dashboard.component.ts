import { Component, OnInit } from '@angular/core';
import { RouterLink } from '@angular/router';
import { DatePipe } from '@angular/common';
import { ApiService } from '../../core/services/api.service';
import { StatusBadgeComponent } from '../../shared/components/status-badge/status-badge.component';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [RouterLink, StatusBadgeComponent, DatePipe],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit {
  backendStatus = 'checking';
  deviceInfo = '';
  mongoStatus = false;
  totalLogs = 0;
  totalSimulations = 0;
  recentLogs: any[] = [];

  constructor(private api: ApiService) {}

  ngOnInit(): void {
    this.loadHealth();
    this.loadStats();
  }

  private loadHealth(): void {
    this.api.getHealth().subscribe({
      next: (health) => {
        this.backendStatus = 'online';
        this.deviceInfo = health.device;
        this.mongoStatus = health.mongodb;
      },
      error: () => {
        this.backendStatus = 'offline';
      }
    });
  }

  private loadStats(): void {
    this.api.getLogs({ limit: 5 }).subscribe({
      next: (data) => {
        this.totalLogs = data.total;
        this.recentLogs = data.logs;
      },
      error: () => {}
    });

    this.api.getSimulations().subscribe({
      next: (data) => {
        this.totalSimulations = data.simulations?.length || 0;
      },
      error: () => {}
    });
  }
}
