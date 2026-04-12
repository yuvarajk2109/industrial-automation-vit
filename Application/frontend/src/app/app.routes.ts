import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
  {
    path: 'dashboard',
    loadComponent: () => import('./pages/dashboard/dashboard.component').then(m => m.DashboardComponent)
  },
  {
    path: 'analyse',
    loadComponent: () => import('./pages/single-analysis/single-analysis.component').then(m => m.SingleAnalysisComponent)
  },
  {
    path: 'simulation',
    loadComponent: () => import('./pages/simulation/simulation.component').then(m => m.SimulationComponent)
  },
  {
    path: 'logs',
    loadComponent: () => import('./pages/logs/logs.component').then(m => m.LogsComponent)
  },
  {
    path: 'finetune',
    loadComponent: () => import('./pages/finetune/finetune.component').then(m => m.FinetuneComponent)
  },
  {
    path: 'about',
    loadComponent: () => import('./pages/about/about.component').then(m => m.AboutComponent)
  },
  { path: '**', redirectTo: 'dashboard' }
];