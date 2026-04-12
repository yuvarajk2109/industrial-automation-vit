import { Component, OnInit, OnDestroy, signal } from '@angular/core';
import { RouterOutlet, RouterLink, RouterLinkActive } from '@angular/router';
import { ApiService } from '../core/services/api.service';

@Component({
  selector: 'app-main-layout',
  standalone: true,
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  templateUrl: './main-layout.component.html',
  styleUrl: './main-layout.component.css'
})
export class MainLayoutComponent implements OnInit, OnDestroy {
  sidebarCollapsed = false;
  backendOnline = false;
  pendingCorrections = 0;
  private healthInterval: any;

  navItems = [
    {
      path: '/dashboard',
      label: 'Dashboard',
      exact: true,
      icon: `<i class="fa-solid fa-chart-line"></i>`
    },
    {
      path: '/analyse',
      label: 'Analyse Image',
      exact: false,
      icon: `<i class="fa-solid fa-microscope"></i>`
    },
    {
      path: '/simulation',
      label: 'Virtual Simulation',
      exact: false,
      icon: `<i class="fa-solid fa-industry"></i>`
    },
    {
      path: '/logs',
      label: 'Analysis Logs',
      exact: false,
      icon: `<i class="fa-solid fa-list-check"></i>`
    },
    {
      path: '/finetune',
      label: 'Model Training',
      exact: false,
      icon: `<i class="fa-solid fa-brain"></i>`
    },
    {
      path: '/about',
      label: 'About',
      exact: false,
      icon: `<i class="fa-solid fa-circle-info"></i>`
    }
  ];

  constructor(private api: ApiService) { }

  ngOnInit(): void {
    this.checkHealth();
    this.loadPendingCorrections();
    this.healthInterval = setInterval(() => {
      this.checkHealth();
      this.loadPendingCorrections();
    }, 15000);
  }

  ngOnDestroy(): void {
    if (this.healthInterval) clearInterval(this.healthInterval);
  }

  toggleSidebar(): void {
    this.sidebarCollapsed = !this.sidebarCollapsed;
  }

  private checkHealth(): void {
    this.api.getHealth().subscribe({
      next: () => (this.backendOnline = true),
      error: () => (this.backendOnline = false)
    });
  }

  private loadPendingCorrections(): void {
    this.api.getFeedbackStats().subscribe({
      next: (stats: any) => {
        this.pendingCorrections = stats?.pending || 0;
      },
      error: () => { }
    });
  }
}