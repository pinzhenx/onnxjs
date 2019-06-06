import {Attribute} from '../../attribute';
import {Operator} from '../../operators';
import {Tensor} from '../../tensor';
import {WebNNInferenceHandler} from './inference-handler';
import {NNSubgraphNode} from './nn-subgraph-node';
import {Graph} from '../../graph';

export class NNSubgraphOp implements Operator {

  public nodes: ReadonlyArray<Graph.Node>;
  public inputs: number[];
  public outputs: number[];

  constructor(nnSubgraphNode: NNSubgraphNode) {
    this.nodes = nnSubgraphNode.nodes;
    this.inputs = nnSubgraphNode.inputs;
    this.outputs = nnSubgraphNode.outputs;
  }

  async run(inferenceHandler: WebNNInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {
    return await inferenceHandler.run(this, inputs);
  };

  initialize(attributes: Attribute): void {}

  checkInputs(inputs: Tensor[]): boolean { return true; }
}
