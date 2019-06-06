// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {OpSet} from '../../opset';
import {NNSubgraphOp} from './nn-subgraph-op';
import {NNSubgraphNode} from './nn-subgraph-node';

export const WEBNN_OP_RESOLVE_RULES: ReadonlyArray<OpSet.ResolveRule> = [
  ['NNSubgraph', '', '1+', (node) => new NNSubgraphOp(node as NNSubgraphNode)],
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
  'Reshape'
];
