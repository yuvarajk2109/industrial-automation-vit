import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'markdown',
  standalone: true
})
export class MarkdownPipe implements PipeTransform {
  transform(value: string): string {
    if (!value) return '';

    let html = value;

    // Headers: ### Title
    html = html.replace(/^### (.*$)/gim, '<h4 class="mt-2 mb-1" style="font-weight: 600;">$1</h4>');
    html = html.replace(/^## (.*$)/gim, '<h3 class="mt-2 mb-1" style="font-weight: 600;">$1</h3>');
    html = html.replace(/^# (.*$)/gim, '<h2 class="mt-2 mb-1" style="font-weight: 600;">$1</h2>');

    // Bold: **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic: *text* (must run after bold)
    html = html.replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

    // Bullet points: * item or - item
    html = html.replace(/^\s*[\*\-] (.*$)/gim, '<li>$1</li>');

    // Wrap adjacent bullet points in an unordered list
    html = html.replace(/(<li>.*?<\/li>\n*)+/gi, (match) => {
      const items = match.trim();
      return `<ul style="padding-left: 1.5rem; margin-top: 0.5rem; margin-bottom: 0.5rem; list-style-type: disc;">${items}</ul>`;
    });

    // Paragraphs / Newlines
    // First, temporarily remove newlines that are next to block elements we just created
    html = html.replace(/<\/h[234]>\n+/g, '</h$1>');
    html = html.replace(/<\/ul>\n+/g, '</ul>');
    
    html = html.replace(/\n\n/g, '<br><br>');
    html = html.replace(/\n(?!(<ul|<li|<h))/g, '<br>');

    return html;
  }
}
