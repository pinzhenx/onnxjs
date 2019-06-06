import {Attribute} from '../../attribute';
import {Operator} from '../../operators';
import {Tensor} from '../../tensor';
import {WebNNInferenceHandler} from './inference-handler';
import {Graph} from '../../graph';
import {TensorUtil} from '../../util';
import {WebNNGraphNode} from './webnn-graph-node';
import {WebNNGraph} from './webnn-graph';

export class WebNNGraphOp implements Operator {

  public nodes: ReadonlyArray<Graph.Node>;
  public inputs: number[];
  public outputs: number[];
  private webnnGraph: WebNNGraph;

  constructor(webnnGraphNode: WebNNGraphNode) {
    this.nodes = webnnGraphNode.nodes;
    this.inputs = webnnGraphNode.inputs;
    this.outputs = webnnGraphNode.outputs;
  }

  initialize(attributes: Attribute): void {
  }

  // TODO: check input tensors
  checkInputs(inputs: Tensor[]): boolean { return true; }

  async run(handler: WebNNInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {
    const nhwcInputs = inputs.map((tensor) => TensorUtil.toNHWC(tensor));
    if (this.webnnGraph === undefined) {
      this.webnnGraph = new WebNNGraph();
      await this.webnnGraph.compile(handler, this, nhwcInputs);
    }
    const nhwcOutputs = await this.webnnGraph.run(handler, nhwcInputs);
    return nhwcOutputs.map((tensor) => TensorUtil.toNCHW(tensor));
  };
}
