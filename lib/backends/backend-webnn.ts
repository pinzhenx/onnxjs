// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {Backend as BackendInterface} from '../api/onnx';
import {Backend, SessionHandler} from '../backend';
import {Logger} from '../instrument';
import {Session} from '../session';

import {WebNNSessionHandler} from './webnn/session-handler';
import {NeuralNetworkContext, NNNavigator} from './webnn/types';

type WebNNOptions = BackendInterface.WebNNOptions;

/**
 * WebGLBackend is the entry point for all WebGL opeartions
 * When it starts it created the WebGLRenderingContext
 * and other main framework components such as Program and Texture Managers
 */
export class WebNNBackend implements Backend, WebNNOptions {
  disabled?: boolean;
  prefer: 'fast'|'sustained'|'low';
  nnContext: NeuralNetworkContext;

  constructor() {
    this.prefer = 'fast';
  }

  initialize(): boolean {
    try {
      this.nnContext = (navigator as NNNavigator).ml.getNeuralNetworkContext();
      Logger.verbose('WebNNBackend', `Created WebGLContext: ${typeof this.nnContext}`);
      return true;
    } catch (e) {
      Logger.warning('WebNNBackend', `Unable to initialize WebNNBackend. ${e}`);
      return false;
    }
  }

  createSessionHandler(context: Session.Context): SessionHandler {
    return new WebNNSessionHandler(this, context);
  }

  dispose() {}
}
