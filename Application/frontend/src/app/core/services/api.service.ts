import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { PipelineResult, HealthCheck } from '../models/prediction.model';
import { ChatResponse } from '../models/chat.model';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly baseUrl = 'http://localhost:5000/api';

  constructor(private http: HttpClient) { }

  getHealth(): Observable<HealthCheck> {
    return this.http.get<HealthCheck>(`${this.baseUrl}/health`);
  }

  predict(imageInput: string | File, domain: string): Observable<PipelineResult> {
    if (imageInput instanceof File) {
      const formData = new FormData();
      formData.append('image', imageInput);
      formData.append('domain', domain);
      return this.http.post<PipelineResult>(`${this.baseUrl}/predict`, formData);
    } else {
      return this.http.post<PipelineResult>(`${this.baseUrl}/predict`, {
        image_path: imageInput,
        domain
      });
    }
  }

  browse(type: 'file' | 'directory' = 'directory'): Observable<{ path: string }> {
    return this.http.get<{ path: string }>(`${this.baseUrl}/browse?type=${type}`);
  }

  chat(logId: string, message: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.baseUrl}/chat`, {
      log_id: logId,
      message
    });
  }

  getLogs(params: {
    session_id?: string;
    domain?: string;
    page?: number;
    limit?: number;
  } = {}): Observable<any> {
    const queryParams: any = {};
    if (params.session_id) queryParams.session_id = params.session_id;
    if (params.domain) queryParams.domain = params.domain;
    if (params.page) queryParams.page = params.page.toString();
    if (params.limit) queryParams.limit = params.limit.toString();

    return this.http.get(`${this.baseUrl}/logs`, { params: queryParams });
  }

  getLogDetail(logId: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/logs/${logId}`);
  }

  getSimulations(): Observable<any> {
    return this.http.get(`${this.baseUrl}/simulations`);
  }

  getStats(): Observable<any> {
    return this.http.get(`${this.baseUrl}/logs/stats`);
  }

  getOutputImageUrl(filename: string): string {
    return `${this.baseUrl}/images/${filename}`;
  }

  getSourceImageUrl(path: string): string {
    return `${this.baseUrl}/source-image?path=${encodeURIComponent(path)}`;
  }

  submitFeedback(feedback: {
    log_id: string;
    domain: string;
    corrected_label: any;
    reason?: string;
  }): Observable<{ feedback_id: string; pending_count: number }> {
    return this.http.post<{ feedback_id: string; pending_count: number }>(
      `${this.baseUrl}/feedback`, feedback
    );
  }

  submitBatchFeedback(sessionId: string, corrections: any[]): Observable<any> {
    return this.http.post(`${this.baseUrl}/feedback/batch`, {
      session_id: sessionId,
      corrections
    });
  }

  getFeedbackStats(): Observable<any> {
    return this.http.get(`${this.baseUrl}/feedback/stats`);
  }

  getFeedback(params: {
    domain?: string;
    status?: string;
    page?: number;
    limit?: number;
  } = {}): Observable<any> {
    const queryParams: any = {};
    if (params.domain) queryParams.domain = params.domain;
    if (params.status) queryParams.status = params.status;
    if (params.page) queryParams.page = params.page.toString();
    if (params.limit) queryParams.limit = params.limit.toString();
    return this.http.get(`${this.baseUrl}/feedback`, { params: queryParams });
  }

  startFineTune(domain: string, config?: any): Observable<any> {
    return this.http.post(`${this.baseUrl}/finetune/start`, { domain, config });
  }

  getFineTuneStatus(): Observable<any> {
    return this.http.get(`${this.baseUrl}/finetune/status`);
  }

  getFineTuneHistory(): Observable<any> {
    return this.http.get(`${this.baseUrl}/finetune/history`);
  }

  getModelVersions(domain?: string): Observable<any> {
    const params = domain ? `?domain=${domain}` : '';
    return this.http.get(`${this.baseUrl}/finetune/versions${params}`);
  }

  rollbackModel(domain: string, version: number): Observable<any> {
    return this.http.post(`${this.baseUrl}/finetune/rollback`, { domain, version });
  }
}