import { Component, OnInit } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../core/services/api.service';
import { BaseChartDirective } from 'ng2-charts';
import { ChartConfiguration, ChartType, Chart, registerables } from 'chart.js';

Chart.register(...registerables);

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [RouterLink, BaseChartDirective],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css'
})
export class DashboardComponent implements OnInit {
  totalLogs = 0;
  totalSimulations = 0;
  totalSteel = 0;
  totalSugar = 0;

  domainChartData: ChartConfiguration['data'] = { labels: [], datasets: [] };
  domainChartOptions: ChartConfiguration['options'] = { responsive: true, maintainAspectRatio: false };
  domainChartType: ChartType = 'doughnut';

  steelChartData: ChartConfiguration['data'] = { labels: [], datasets: [] };
  steelChartOptions: ChartConfiguration['options'] = { responsive: true, maintainAspectRatio: false, indexAxis: 'y' };
  steelChartType: ChartType = 'bar';

  sugarChartData: ChartConfiguration['data'] = { labels: [], datasets: [] };
  sugarChartOptions: ChartConfiguration['options'] = { responsive: true, maintainAspectRatio: false };
  sugarChartType: ChartType = 'bar';

  perfChartData: ChartConfiguration['data'] = { labels: [], datasets: [] };
  perfChartOptions: ChartConfiguration['options'] = { responsive: true, maintainAspectRatio: false };
  perfChartType: ChartType = 'line';

  constructor(private api: ApiService) { }

  ngOnInit(): void {
    this.loadStats();
    this.loadCharts();
  }

  private loadStats(): void {
    this.api.getLogs({ limit: 1 }).subscribe({
      next: (data) => {
        this.totalLogs = data.total || 0;
      },
      error: () => { }
    });

    this.api.getLogs({ domain: 'steel', limit: 1 }).subscribe({
      next: (data) => {
        this.totalSteel = data.total || 0;
      },
      error: () => { }
    });

    this.api.getLogs({ domain: 'sugar', limit: 1 }).subscribe({
      next: (data) => {
        this.totalSugar = data.total || 0;
      },
      error: () => { }
    });

    this.api.getSimulations().subscribe({
      next: (data) => {
        this.totalSimulations = data.simulations?.length || 0;
      },
      error: () => { }
    });
  }

  private loadCharts(): void {
    this.api.getStats().subscribe({
      next: (data) => {
        this.domainChartData = {
          labels: data.domains.map((d: any) => d._id === 'steel' ? 'Steel' : 'Sugar'),
          datasets: [{ data: data.domains.map((d: any) => d.count), backgroundColor: ['#607D8B', '#FF9800'] }]
        };

        this.steelChartData = {
          labels: data.steel_decisions.map((d: any) => (d._id || 'Unknown').replace(/_/g, ' ')),
          datasets: [{ label: 'Frequency', data: data.steel_decisions.map((d: any) => d.count), backgroundColor: '#607D8B' }]
        };

        this.sugarChartData = {
          labels: data.sugar_classes.map((d: any) => (d._id || 'Unknown').replace(/_/g, ' ')),
          datasets: [{ label: 'Frequency', data: data.sugar_classes.map((d: any) => d.count), backgroundColor: '#FF9800' }]
        };

        this.perfChartData = {
          labels: data.performance.map((_: any, i: number) => `#${i + 1}`),
          datasets: [{
            label: 'Processing Time (ms)',
            data: data.performance.map((d: any) => d.processing_time_ms),
            borderColor: '#4CAF50',
            backgroundColor: 'rgba(76, 175, 80, 0.2)',
            fill: true,
            tension: 0.4
          }]
        };
      }
    });
  }
}