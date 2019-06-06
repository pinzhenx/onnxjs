import {Attribute} from '../../attribute';
import {Graph} from '../../graph';

export class WebNNGraphNode implements Graph.Node {
  constructor(public nodes: ReadonlyArray<Graph.Node>,
              public inputs: number[],
              public outputs: number[]) {
    this.name = `${this.graphSummary()} (${this.hashCode()})`;
    this.opType = 'WebNNGraph';
    this.attributes = new Attribute(null);
    this.executeNode = true;
  }

  name: string;
  opType: string;
  attributes: Attribute;
  executeNode: boolean;

  graphSummary() {
    const Object_entries = (o: any) => Object.keys(o).map(k => [k, o[k]]); // polyfill for Object.entries
    return Object_entries(
              this.nodes
                .map((node) => node.opType)
                .reduce((cnt: any, t: any) => {cnt[t] ? cnt[t]++ : cnt[t]=1; return cnt}, {})
           ).map((n: any) => `${n[0]} x ${n[1]}`)
            .join(', ');
  }

  hashCode() {
    return (Array.from(JSON.stringify(this))
        .reduce((s, c) => Math.imul(31, s) + c.charCodeAt(0) | 0, 0) + 2 ** 31).toString(16)
  }
}
