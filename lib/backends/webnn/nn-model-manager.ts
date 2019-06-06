// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Logger, Profiler} from '../../instrument';
import {NeuralNetworkContext} from './types';
import {Tensor} from '../../tensor';
import {NNModel} from './nn-model';
import {NNSubgraphOp} from './nn-subgraph-op';
import {WebNNInferenceHandler} from './inference-handler';

/**
 * NNModelManager is the factory class for WebNN model
 * It compiles NNSubgraphOp into NNModel
 */
export class NNModelManager {

  repo: Map<NNSubgraphOp, NNModel>;  // this should be per-session object

  constructor(public profiler: Readonly<Profiler>, public nnContext: NeuralNetworkContext) {
    this.repo = new Map();
  }
  async getCompiledModel(op: NNSubgraphOp): Promise<NNModel|undefined> {
    return this.repo.get(op);
  }
  async createCompiledModel(handler: WebNNInferenceHandler, op: NNSubgraphOp, inputs: Tensor[]): Promise<NNModel> {
    return await this.profiler.event('backend', 'NNModelManager.createCompiledModel', async () => {
      try {
        return await new NNModel(this.nnContext).compile(handler, op, inputs);
      } catch (err) {
        Logger.error('NNModelManager', '');
        throw err;
      }
    });
  }
  setCompiledModel(key: NNSubgraphOp, compiledModel: NNModel): void {
    this.repo.set(key, compiledModel);
  }
  async run(handler: WebNNInferenceHandler, model: NNModel, inputs: Tensor[]): Promise<Tensor[]> {
    return await model.run(handler, inputs);
  }
  dispose(): void {
    this.repo.clear();
  }
}
