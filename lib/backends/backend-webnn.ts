// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend as BackendInterface} from '../api/onnx';
import {Backend, SessionHandler} from '../backend';
import {WasmBackend} from './backend-wasm';
import {Logger} from '../instrument';
import {Session} from '../session';

import {WebNNSessionHandler} from './webnn/session-handler';
import {NeuralNetworkContext, NNNavigator, preferStrType} from './webnn/types';

type WebNNOptions = BackendInterface.WebNNOptions;

export class WebNNBackend implements Backend, WebNNOptions {
  disabled?: boolean;
  prefer: preferStrType;
  wasmFallback: boolean;
  cpuFallback: boolean;
  nnContext: NeuralNetworkContext;

  constructor() {
    this.prefer = 'fast';
    this.wasmFallback = true;
    this.cpuFallback = true;
  }

  async initialize(): Promise<boolean> {
    try {
      this.nnContext = (navigator as NNNavigator).ml.getNeuralNetworkContext();
      Logger.verbose('WebNNBackend', `Acquired WebNNContext: ${typeof this.nnContext}`);

      if (this.wasmFallback) {
        await new WasmBackend().initialize();
      }

      return true;
    } catch (e) {
      Logger.warning('WebNNBackend', `Unable to initialize WebNNBackend. ${e}`);
      return false;
    }
  }

  createSessionHandler(context: Session.Context): SessionHandler {
    return new WebNNSessionHandler(this, context, this.wasmFallback, this.cpuFallback, this.prefer);
  }

  dispose() {}
}
