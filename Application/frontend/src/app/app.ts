import { Component } from '@angular/core';
import { MainLayoutComponent } from './layout/main-layout.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [MainLayoutComponent],
  template: `<app-main-layout />`
})
export class App { }