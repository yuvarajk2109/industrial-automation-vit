import { Component, Input, OnChanges } from '@angular/core';

interface KGNode {
  id: string;
  x: number;
  y: number;
  label: string;
  isActive: boolean;
  nodeType?: string;
}

interface KGEdge {
  from: string;
  to: string;
  isActive: boolean;
}

@Component({
  selector: 'app-kg-graph',
  standalone: true,
  templateUrl: './kg-graph.component.html',
  styleUrl: './kg-graph.component.css'
})
export class KgGraphComponent implements OnChanges {
  @Input() activatedNodes: string[] = [];
  @Input() traversalPath: any[] = [];
  @Input() domain: string = 'steel';

  nodes: KGNode[] = [];
  edges: KGEdge[] = [];
  svgWidth = 600;
  svgHeight = 300;

  ngOnChanges(): void {
    this.buildGraph();
  }

  private buildGraph(): void {
    if (this.domain === 'steel') {
      this.buildSteelGraph();
    } else {
      this.buildSugarGraph();
    }
  }

  private buildSteelGraph(): void {
    const layers = [
      ['Defect_Class_1_Detected', 'Defect_Class_2_Detected', 'Defect_Class_3_Detected', 'Defect_Class_4_Detected'],
      ['Defect_Area', 'Defect_Length', 'Defect_Density', 'Defect_Distribution'],
      ['Isolated_Minor_Defect', 'Localized_Severe_Defect', 'Widespread_Defect_Pattern', 'Critical_Structural_Defect'],
      ['Acceptable_Quality', 'Marginal_Quality', 'Unacceptable_Quality'],
      ['Accept_Strip', 'Downgrade_Strip', 'Reject_Strip', 'Manual_Inspection_Required']
    ];

    this.svgWidth = 700;
    this.svgHeight = 320;
    this.nodes = [];
    this.edges = [];

    const xSpacing = this.svgWidth / (layers.length + 1);

    layers.forEach((layer, li) => {
      const ySpacing = this.svgHeight / (layer.length + 1);
      layer.forEach((nodeId, ni) => {
        this.nodes.push({
          id: nodeId,
          x: xSpacing * (li + 1),
          y: ySpacing * (ni + 1),
          label: nodeId.replace(/_/g, ' ').replace('Detected', '').trim(),
          isActive: this.activatedNodes.includes(nodeId)
        });
      });
    });

    // Build edges from traversal path
    for (const tp of this.traversalPath) {
      this.edges.push({
        from: tp.from,
        to: tp.to,
        isActive: true
      });
    }
  }

  private buildSugarGraph(): void {
    const stateNodes = ['UNSATURATED', 'METASTABLE', 'INTERMEDIATE', 'LABILE'];
    const condNodes = ['low_nucleation_risk', 'medium_nucleation_risk', 'high_nucleation_risk', 'stable_growth', 'unstable_growth'];
    const actionNodes = ['hold_process', 'increase_evaporation', 'reduce_seeding', 'emergency_dilution'];

    this.svgWidth = 600;
    this.svgHeight = 300;
    this.nodes = [];
    this.edges = [];

    // States column
    stateNodes.forEach((id, i) => {
      this.nodes.push({ id, x: 100, y: 50 + i * 60, label: id.toLowerCase(), isActive: this.activatedNodes.includes(id) });
    });

    // Conditions column
    condNodes.forEach((id, i) => {
      this.nodes.push({ id, x: 300, y: 30 + i * 55, label: id.replace(/_/g, ' '), isActive: this.activatedNodes.includes(id) });
    });

    // Actions column
    actionNodes.forEach((id, i) => {
      this.nodes.push({ id, x: 500, y: 50 + i * 65, label: id.replace(/_/g, ' '), isActive: this.activatedNodes.includes(id) });
    });

    for (const tp of this.traversalPath) {
      this.edges.push({ from: tp.from, to: tp.to, isActive: true });
    }
  }

  getNode(id: string): KGNode | undefined {
    return this.nodes.find(n => n.id === id);
  }
}
