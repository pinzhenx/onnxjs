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

import {preferStrType, NeuralNetworkContext} from './types';
import {WebNNInferenceHandler} from './inference-handler';
import {WEBNN_OP_RESOLVE_RULES, WEBNN_SUPPORTED_OPS} from './op-resolve-rules';
import {CPU_OP_RESOLVE_RULES} from '../cpu/op-resolve-rules';
import {WASM_OP_RESOLVE_RULES} from '../wasm/op-resolve-rules';

import {GraphUtils} from './graph-utils';
import {WebNNGraphNode} from './webnn-graph-node';

export class WebNNSessionHandler implements SessionHandler, Graph.Initializer {
  webnnContext: NeuralNetworkContext;
  private tensorCache: Map<number, Tensor>;
  private initializers: Set<number>;
  private opResolveRules: ReadonlyArray<OpSet.ResolveRule>;

  private graph: Graph;

  constructor(public readonly backend: WebNNBackend, public readonly context: Session.Context,
              public readonly wasmFallback: boolean, public readonly cpuFallback: boolean,
              public readonly prefer: preferStrType) {
    this.webnnContext = backend.nnContext;
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

  // It groups the WebNN-supported nodes into a fused single node of type `WebNNGraph`,
  // and then rewrite the ingoing and outgoing edges of each tensor (Graph.Value).
  // Ideally this should not be done inside the session-handler, but the framework hasn't
  // supported graph partitioning and nested subgraph yet. So we temporarily transform the
  // graph here and this implementation still has some known issues, e.g., Graph.finalizeGraph()
  // will break some graphs
  transformGraph(transformer: Graph.Transformer) {

    const nnSupportedOps = new Set(WEBNN_SUPPORTED_OPS);

    const graphNodes = transformer.getNodes();
    const newGraphNodes = [];
    const colorGraph = new GraphUtils(graphNodes.length);

    graphNodes.forEach((op, i) => {
      colorGraph.addNode(i, op.inputs, op.outputs);
      if (!nnSupportedOps.has(op.opType)) {
        colorGraph.setBlack(i); // mark unsupported ops black
      }
    });
    colorGraph.identifyInputOutputTensors(transformer.getInputIndices(), transformer.getOutputIndices());
    for (const {nodeIds, inputIds, outputIds, color} of colorGraph.partition()) {
      const nodes = nodeIds.map((id) => graphNodes[id]);
      if (color === 'black') {
        newGraphNodes.push(...nodes);
      } else {
        newGraphNodes.push(new WebNNGraphNode(nodes, inputIds, outputIds));
      }
    }
    transformer.setNodes(newGraphNodes);

    // re-link all tensors to new nodes
    const oldNameToNewName: Map<string, string> = new Map();
    for (const node of newGraphNodes) {
      if (node.opType === 'WebNNGraph') {
        const outerGraphName = node.name;
        for (const innerNode of (node as WebNNGraphNode).nodes) {
          oldNameToNewName.set(innerNode.name, outerGraphName);
        }
      } else {
        oldNameToNewName.set(node.name, node.name);
      }
    }

    const newNameToNewId: Map<string, number> = new Map();
    for (const [i, node] of newGraphNodes.entries()) {
      newNameToNewId.set(node.name, i);
    }

    const graphValues= transformer.getValues();
    const newGraphValues = [];
    for (const value of graphValues) {
      let oldFromId = value.from;
      let newFromId = oldFromId;
      if (oldFromId !== undefined && oldFromId !== -1) {
        let oldFromName = graphNodes[oldFromId].name;
        let newFromName = oldNameToNewName.get(oldFromName)!;
        newFromId = newNameToNewId.get(newFromName)!;
      }

      let oldToIds = value.to;
      let oldToNames= oldToIds.map((oldId) => graphNodes[oldId].name);
      let newToNames = oldToNames.map((oldName) => oldNameToNewName.get(oldName)!);
      let newToIds = newToNames.map((newName) => newNameToNewId.get(newName)!);

      let newValue = transformer.makeNewValue(newFromId, newToIds, value.tensor, value.type);
      newGraphValues.push(newValue);
    }

    transformer.setValues(newGraphValues);
  }

  createInferenceHandler() {
    return new WebNNInferenceHandler(this);
  }

  onGraphInitialized(graph: Graph): void {
    this.graph = graph;
    for (const [tensorId, v] of graph.getValues().entries()) {
      if (v.from === -1 && v.tensor) {
        this.initializers.add(tensorId);
      }
    }
  }

  isInitializer(tensorId: number): boolean {
    return this.initializers ? this.initializers.has(tensorId) : false;
  }

  /**
   * return tensor in NHWC format
   */
  getTensor(tensorId: number): Tensor {
    let tensor;
    if (this.isInitializer(tensorId)) {
      const value = this.graph.getValues()[tensorId];
      const nchwTensor = value && value.tensor!;
      tensor = nchwTensor && TensorUtil.toNHWC(nchwTensor);
    } else {
      tensor = this.tensorCache.get(tensorId);
    }
    if (tensor === undefined) {
      throw new Error(`Cannot find tensor ${tensorId}`);
    }
    return tensor;
  }

  /**
   * add tensor in NHWC format
   */
  setTensor(id: number, tensor: Tensor) {
    this.tensorCache.set(id, tensor);
  }
  dispose(): void {
    this.tensorCache.clear();
    this.initializers.clear();
  }
  resolve(node: Graph.Node, opsets: ReadonlyArray<OpSet>): Operator {
    const op = resolveOperator(node, opsets, this.opResolveRules);
    op.initialize(node.attributes);
    return op;
  }
}
