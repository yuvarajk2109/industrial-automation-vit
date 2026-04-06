import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { PipelineResult, HealthCheck } from '../models/prediction.model';
import { ChatResponse } from '../models/chat.model';

@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly baseUrl = 'http://localhost:5000/api';

  constructor(private http: HttpClient) {}

  /** Health check */
  getHealth(): Observable<HealthCheck> {
    return this.http.get<HealthCheck>(`${this.baseUrl}/health`);
  }

  /** Single image analysis */
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

  /** Browse local file/folder using backend native OS dialog */
  browse(type: 'file' | 'directory' = 'directory'): Observable<{path: string}> {
    return this.http.get<{path: string}>(`${this.baseUrl}/browse?type=${type}`);
  }

  /** Chat with Gemini in context of an analysis */
  chat(logId: string, message: string): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.baseUrl}/chat`, {
      log_id: logId,
      message
    });
  }

  /** Retrieve paginated logs */
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

  /** Retrieve a single log by ID */
  getLogDetail(logId: string): Observable<any> {
    return this.http.get(`${this.baseUrl}/logs/${logId}`);
  }

  /** Retrieve all simulation summaries */
  getSimulations(): Observable<any> {
    return this.http.get(`${this.baseUrl}/simulations`);
  }

  /** Retrieve aggregated chart stats */
  getStats(): Observable<any> {
    return this.http.get(`${this.baseUrl}/logs/stats`);
  }

  /** Get the URL for an output image */
  getOutputImageUrl(filename: string): string {
    return `${this.baseUrl}/images/${filename}`;
  }

  /** Get the URL for a source image */
  getSourceImageUrl(path: string): string {
    return `${this.baseUrl}/source-image?path=${encodeURIComponent(path)}`;
  }
}
