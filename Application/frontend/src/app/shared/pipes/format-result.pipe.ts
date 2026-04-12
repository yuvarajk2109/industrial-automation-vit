import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'formatResult',
  standalone: true
})
export class FormatResultPipe implements PipeTransform {
  transform(value: any): string {
    if (typeof value !== 'string' || !value) return value || '-';
    return value
      .replace(/_/g, ' ')
      .split(' ')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
      .join(' ');
  }
}