// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceHandler} from '../../backend';
import {Tensor} from '../../tensor';
import {NNSubgraphOp} from './nn-subgraph-op';
import {WebNNSessionHandler} from './session-handler';
import {TensorUtil} from '../../util';

export class WebNNInferenceHandler implements InferenceHandler {

  constructor(public session: WebNNSessionHandler) {}

  async run(op: NNSubgraphOp, inputs: Tensor[]): Promise<Tensor[]> {
    const nhwcInputs = inputs.map((tensor) => TensorUtil.toNHWC(tensor));
    let model = await this.session.nnModelManager.getCompiledModel(op);
    if (model === undefined) {
      model = await this.session.nnModelManager.createCompiledModel(this, op, nhwcInputs);
      this.session.nnModelManager.setCompiledModel(op, model);
    }
    const nhwcOutputs = await this.session.nnModelManager.run(this, model, nhwcInputs);
    return nhwcOutputs.map((tensor) => TensorUtil.toNCHW(tensor));
  }

  getTensor(id: number): Tensor {
    return this.session.getTensor(id);
  }

  setTensor(id: number, tensor: Tensor) {
    this.session.setTensor(id, tensor);
  }

  dispose(): void {
  }

}
