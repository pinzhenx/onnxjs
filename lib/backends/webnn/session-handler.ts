// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

import {SessionHandler} from '../../backend';
import {Graph} from '../../graph';
import {Operator} from '../../operators';
import {OpSet, resolveOperator} from '../../opset';
import {Session} from '../../session';
import {Tensor} from '../../tensor';
import {TensorUtil} from '../../util';
import {WebNNBackend} from '../backend-webnn';

import {preferStrType} from './types';
import {WebNNInferenceHandler} from './inference-handler';
import {WEBNN_OP_RESOLVE_RULES, WEBNN_SUPPORTED_OPS} from './op-resolve-rules';
import {CPU_OP_RESOLVE_RULES} from '../cpu/op-resolve-rules';
import {WASM_OP_RESOLVE_RULES} from '../wasm/op-resolve-rules';

import {GraphUtils} from './graph-utils';
import {NNSubgraphNode} from './nn-subgraph-node';
import {NNModelManager} from './nn-model-manager';

export class WebNNSessionHandler implements SessionHandler, Graph.Initializer {
  nnModelManager: NNModelManager;
  private tensorCache: Map<number, Tensor>;
  private initializers: Set<number>;
  private opResolveRules: ReadonlyArray<OpSet.ResolveRule>;

  constructor(public readonly backend: WebNNBackend, public readonly context: Session.Context,
              public readonly wasmFallback: boolean, public readonly cpuFallback: boolean,
              public readonly prefer: preferStrType) {
    this.nnModelManager = new NNModelManager(this.context.profiler, backend.nnContext);
    this.tensorCache = new Map();
    this.initializers = new Set();
    this.opResolveRules = WEBNN_OP_RESOLVE_RULES;
    if (wasmFallback) {
      this.opResolveRules = this.opResolveRules.concat(WASM_OP_RESOLVE_RULES);
    }
    if (cpuFallback) {
      this.opResolveRules = this.opResolveRules.concat(CPU_OP_RESOLVE_RULES);
    }
  }

  transformGraph(transformer: Graph.Transformer) {

    const nnSupportedOps = new Set(WEBNN_SUPPORTED_OPS);

    const graphNodes = transformer.getNodes();
    const colorGraph = new GraphUtils(graphNodes.length);

    graphNodes.forEach((op, i) => {
      colorGraph.addNode(i, op.inputs, op.outputs);
      if (!nnSupportedOps.has(op.opType)) {
        colorGraph.setBlack(i); // mark unsupported ops black
      }
    });
    colorGraph.identifyInputOutputTensors(transformer.getInputIndices(), transformer.getOutputIndices());
    const newNodes = [];
    for (const {nodeIds, inputIds, outputIds} of colorGraph.partition()) {
      const nodes = nodeIds.map((id) => graphNodes[id]);
      if (nnSupportedOps.has(nodes[0].opType)) {
        newNodes.push(new NNSubgraphNode(nodes, inputIds, outputIds));
      } else {
        newNodes.push(...nodes);
      }
    }
    transformer.setNodes(newNodes);
  }

  createInferenceHandler() {
    return new WebNNInferenceHandler(this);
  }
  onGraphInitialized(graph: Graph): void {
    for (const [tensorId, v] of graph.getValues().entries()) {
      if (v.from === -1 && v.tensor) {
        const tensor = v.tensor;
        this.initializers.add(tensorId);
        this.tensorCache.set(tensorId, TensorUtil.toNHWC(tensor));
      }
    }
  }
  isInitializer(tensorId: number): boolean {
    return this.initializers ? this.initializers.has(tensorId) : false;
  }
  getTensor(tensorId: number): Tensor {
    const tensor = this.tensorCache.get(tensorId);
    if (tensor === undefined) {
      throw new Error(`Cannot find tensor ${tensorId}`);
    } else {
      return tensor;
    }
  }
  setTensor(id: number, tensor: Tensor) {
    this.tensorCache.set(id, tensor);  // add WebNN NHWC tensor
  }
  dispose(): void {
    this.nnModelManager.dispose();
    this.tensorCache.clear();
    this.initializers.clear();
  }
  resolve(node: Graph.Node, opsets: ReadonlyArray<OpSet>): Operator {
    const op = resolveOperator(node, opsets, this.opResolveRules);
    op.initialize(node.attributes);
    return op;
  }
}
