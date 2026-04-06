import { Injectable, NgZone } from '@angular/core';
import { Observable } from 'rxjs';
import { SimulationEvent, SimulationConfig } from '../models/simulation.model';

@Injectable({ providedIn: 'root' })
export class SimulationService {
  private readonly baseUrl = 'http://localhost:5000/api';
  private eventSource: EventSource | null = null;

  constructor(private ngZone: NgZone) {}

  /**
   * Start a simulation and return an observable stream of SSE events.
   * Uses POST to initiate, then switches to EventSource for streaming.
   */
  startSimulation(config: SimulationConfig): Observable<SimulationEvent> {
    return new Observable(observer => {
      // POST to start simulation, which returns SSE stream
      const url = `${this.baseUrl}/simulate`;

      // Use fetch to POST and receive SSE stream
      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
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

          const readStream = (): void => {
            reader.read().then(({ done, value }) => {
              if (done) {
                this.ngZone.run(() => observer.complete());
                return;
              }

              const text = decoder.decode(value, { stream: true });
              const lines = text.split('\n');

              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  try {
                    const data = JSON.parse(line.substring(6));
                    this.ngZone.run(() => observer.next(data));

                    if (data.step === 'simulation_complete') {
                      this.ngZone.run(() => observer.complete());
                      return;
                    }
                  } catch (e) {
                    // Skip malformed JSON lines
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
        // Cleanup on unsubscribe
        this.stopSimulation();
      };
    });
  }

  /** Stop any running simulation */
  stopSimulation(): void {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
    }
  }
}
