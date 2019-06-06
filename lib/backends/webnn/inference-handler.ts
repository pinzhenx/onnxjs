// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {InferenceHandler} from '../../backend';
import {Tensor} from '../../tensor';
import {WebNNSessionHandler} from './session-handler';

export class WebNNInferenceHandler implements InferenceHandler {

  constructor(public session: WebNNSessionHandler) {}

  /**
   * return tensor in NHWC format
   */
  getTensor(id: number): Tensor {
    return this.session.getTensor(id);
  }

  /**
   * add tensor in NHWC format
   */
  setTensor(id: number, tensor: Tensor) {
    this.session.setTensor(id, tensor);
  }

  dispose(): void {
  }

}
