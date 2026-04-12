import { Injectable, NgZone } from '@angular/core';
import { Observable } from 'rxjs';
import { SimulationEvent, SimulationConfig } from '../models/simulation.model';

@Injectable({ providedIn: 'root' })
export class SimulationService {
  private readonly baseUrl = 'http://localhost:5000/api';
  private abortController: AbortController | null = null;

  constructor(private ngZone: NgZone) { }

  startSimulation(config: SimulationConfig): Observable<SimulationEvent> {
    return new Observable(observer => {
      this.abortController = new AbortController();
      const url = `${this.baseUrl}/simulate`;

      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
        signal: this.abortController.signal
      })
        .then(response => {
          if (!response.ok) {
            throw new Error(`Simulation failed: ${response.statusText}`);
          }

          const reader = response.body?.getReader();
          const decoder = new TextDecoder();

          if (!reader) {
            throw new Error('No response body reader available');
          }

          let buffer = '';

          const readStream = (): void => {
            reader.read().then(({ done, value }) => {
              if (done) {
                this.ngZone.run(() => observer.complete());
                return;
              }

              buffer += decoder.decode(value, { stream: true });
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';

              for (const line of lines) {
                if (line.trim() && line.startsWith('data: ')) {
                  try {
                    const data = JSON.parse(line.substring(6));
                    this.ngZone.run(() => observer.next(data));

                    if (data.step === 'simulation_complete') {
                      this.ngZone.run(() => observer.complete());
                      return;
                    }
                  } catch (e) {
                  }
                }
              }

              readStream();
            }).catch(error => {
              this.ngZone.run(() => observer.error(error));
            });
          };

          readStream();
        })
        .catch(error => {
          this.ngZone.run(() => observer.error(error));
        });

      return () => {
        this.stopSimulation();
      };
    });
  }

  stopSimulation(): void {
    if (this.abortController) {
      this.abortController.abort();
      this.abortController = null;
    }
  }
}