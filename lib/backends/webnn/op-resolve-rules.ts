// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {OpSet} from '../../opset';
import {WebNNGraphOp} from './webnn-graph-op';
import {WebNNGraphNode} from './webnn-graph-node';

export const WEBNN_OP_RESOLVE_RULES: ReadonlyArray<OpSet.ResolveRule> = [
  ['WebNNGraph', '', '1+', (node) => new WebNNGraphOp(node as WebNNGraphNode)],
];

export const WEBNN_SUPPORTED_OPS: ReadonlyArray<string> = [
  'Conv',
  'BatchNormalization',
  'Relu',
  'GlobalAveragePool',
  'AveragePool',
  'MaxPool',
  'Concat',
  'Softmax',
  'Add',
  'Mul',
  'Sum',
  'Gemm',
  'Reshape',
  'Dropout', // keep uncommented to prevent Graph.finalizeGraph() from breaking the whole graph
];
