import { Injectable, signal, computed } from '@angular/core';
import { ApiService } from './api.service';
import { ChatMessage } from '../models/chat.model';

@Injectable({ providedIn: 'root' })
export class ChatService {
  messages = signal<ChatMessage[]>([]);
  isLoading = signal(false);
  currentLogId = signal<string | null>(null);

  hasMessages = computed(() => this.messages().length > 0);

  constructor(private api: ApiService) { }

  initializeChat(logId: string, initialResponse: string): void {
    this.currentLogId.set(logId);
    this.messages.set([
      { role: 'model', content: initialResponse }
    ]);
  }

  sendMessage(message: string): void {
    const logId = this.currentLogId();
    if (!logId || !message.trim()) return;

    this.messages.update(msgs => [
      ...msgs,
      { role: 'user', content: message.trim() }
    ]);

    this.isLoading.set(true);

    this.api.chat(logId, message.trim()).subscribe({
      next: (response) => {
        this.messages.update(msgs => [
          ...msgs,
          { role: 'model', content: response.response }
        ]);
        this.isLoading.set(false);
      },
      error: (err) => {
        this.messages.update(msgs => [
          ...msgs,
          {
            role: 'model',
            content: 'Sorry, I encountered an error processing your request. Please try again.'
          }
        ]);
        this.isLoading.set(false);
      }
    });
  }

  clearChat(): void {
    this.messages.set([]);
    this.currentLogId.set(null);
    this.isLoading.set(false);
  }
}