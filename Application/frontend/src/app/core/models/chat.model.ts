export interface ChatMessage {
  role: 'user' | 'model';
  content: string;
  timestamp?: string;
}

export interface ChatRequest {
  log_id: string;
  message: string;
}

export interface ChatResponse {
  response: string;
  history: ChatMessage[];
}