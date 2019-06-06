import {ShapeUtil} from '../../util';
import {Tensor} from '../../tensor';
import {NeuralNetworkContext, NNTensorType, OperandOptions, Model, Compilation, Execution} from './types';
import {WebNNInferenceHandler} from './inference-handler';
import {WebNNGraphOp} from './webnn-graph-op';

export class WebNNGraph {

  constructor() {
    this._operandIndex = 0;
    this._onnxIdToNNId = new Map();
  }

  async run(handler: WebNNInferenceHandler, inputs: Tensor[]): Promise<Tensor[]> {
    // bind input tensors at runtime
    inputs.forEach((tensor, i) => {
      this._execution.setInput(i, tensor.data as NNTensorType);
    });
    await this._execution.startCompute();
    return this._graph.outputs.map((onnxId) => handler.getTensor(onnxId));
  };

  async compile(handler: WebNNInferenceHandler,
                graph: WebNNGraphOp,
                inputs: Tensor[]): Promise<WebNNGraph> {

    this._nn = handler.session.webnnContext;
    this._model = await this._nn.createModel();
    this._graph = graph;

    inputs.forEach((tensor, i) => {
      const nnId = this._addTensorFloat32(Array.from(tensor.dims));
      const onnxId = this._graph.inputs[i];
      this._setOnnxIdToNNId(onnxId, nnId);
      handler.setTensor(onnxId, tensor);
    });

    this._addOpsAndParams(handler);
    this._addInputsOutputs();
    await this._model.finish();

    const preferCodeMap = {
      fast: this._nn.PREFER_FAST_SINGLE_ANSWER,
      sustained: this._nn.PREFER_SUSTAINED_SPEED,
      low: this._nn.PREFER_LOW_POWER,
    };

    const prefer = handler.session.prefer;
    this._compilation = await this._model.createCompilation();
    this._compilation.setPreference(preferCodeMap[prefer]);
    await this._compilation.finish();

    this._execution = await this._compilation.createExecution();
    // allocate and bind output buffers at compile time
    this._graph.outputs.forEach((onnxId, i) => {
      const tensor = handler.getTensor(onnxId);
      const buffer = new Float32Array(ShapeUtil.size(tensor.dims));
      const tensorWithBuffer = new Tensor(tensor.dims, tensor.type, undefined, undefined, buffer);
      handler.setTensor(onnxId, tensorWithBuffer);
      // TODO: remove type casting
      this._execution.setOutput(i, tensorWithBuffer.floatData as NNTensorType);
    });

    return this;
  }

  _getNNTensorId(onnxId: number) {
    const nnId = this._onnxIdToNNId.get(onnxId);
    if (nnId === undefined) {
      throw new Error(`Cannot find onnx tensor ${onnxId}`);
    }
    return nnId;
  }

  _setOnnxIdToNNId(onnxId: number, nnId: number) {
    this._onnxIdToNNId.set(onnxId, nnId);
  }

  _addInputsOutputs() {
    const modelInputs = this._graph.inputs.map((onnxId) => this._getNNTensorId(onnxId));
    const modelOutputs = this._graph.outputs.map((onnxId) => this._getNNTensorId(onnxId));
    this._model.identifyInputsAndOutputs(modelInputs, modelOutputs);
  }

  _addOpsAndParams(handler: WebNNInferenceHandler) {

    for (let i = 0; i < this._graph.nodes.length; i++) {

      let opType: number = -1;
      let inputs: number[] = [];
      let outputs: number[] = [];

      let node = this._graph.nodes[i];
      let attributes = node.attributes;

      switch(node.opType) {
        case 'Conv': {
          const convNode = node;
          const input = handler.getTensor(node.inputs[0]);
          const convFilter = handler.getTensor(node.inputs[1]);
          const convBias = node.inputs[2] !== undefined ? handler.getTensor(node.inputs[2]) : undefined;

          const nGroups = attributes.getInt('group', 1);
          const convFilterDims = Array.from(convFilter.dims);
          const nChannels = convFilterDims[0];
          let convFilterTensor = convFilter.floatData as Float32Array;
          const convBiasTensor = convBias !== undefined ?  // optional bias
              convBias.floatData as Float32Array :
              new Float32Array(nChannels).fill(0);

          const kernelShape = attributes.getInts('kernel_shape', []);
          if (!kernelShape || kernelShape.length !== 2) {
            throw new Error('Invalid kernelShape');
          }
          const kernelHeight = kernelShape[0];
          const kernelWidth = kernelShape[1];

          const autoPad = attributes.getString('auto_pad', 'NOTSET');
          if (autoPad === 'SAME_UPPER' || autoPad === 'SAME_LOWER') {
            throw new Error('Asymmetric padding is not supported.');
          }

          const pads = attributes.getInts('pads', [0, 0, 0, 0]);
          if (pads.length !== 4) {
            throw new Error('Invalid pads');
          }
          const paddingHeightBegin = pads[0];
          const paddingWidthBegin = pads[1];
          const paddingHeightEnd = pads[2];
          const paddingWidthEnd = pads[3];

          const strides = attributes.getInts('strides', [1, 1]);
          if (!strides || strides.length !== 2) {
            throw new Error('Invalid strides');
          }
          const strideY = strides[0];
          const strideX = strides[1];

          let nextNode = this._graph.nodes[i + 1];
          // fuse batch norm preceded by a conv
          if (nextNode &&
              nextNode.opType === 'BatchNormalization' &&
              node.outputs[0] === nextNode.inputs[0]) {
            const bnNode = nextNode;
            const scale = handler.getTensor(bnNode.inputs[1]);
            const bnBias = handler.getTensor(bnNode.inputs[2]);
            const mean = handler.getTensor(bnNode.inputs[3]);
            const variance = handler.getTensor(bnNode.inputs[4]);
            const epsilon = bnNode.attributes.getFloat('epsilon', 1e-5);

            const scaleTensor = scale.floatData;
            const meanTensor = mean.floatData;
            const varTensor = variance.floatData;
            const bnBiasTensor = bnBias.floatData;

            const nPixels = ShapeUtil.size(convFilterDims.slice(1));
            for (let c = 0; c < nChannels; c++) {
              const w = scaleTensor[c] / Math.sqrt(varTensor[c] + epsilon);
              convBiasTensor[c] = bnBiasTensor[c] + (convBiasTensor[c] - meanTensor[c]) * w;
              for (let p = c * nPixels; p < (c+1) * nPixels; p++) {
                convFilterTensor[p] *= w;
              }
            }

            i++;
            node = nextNode;
            nextNode = this._graph.nodes[i + 1];
          }

          // reshape kernel for depthwise conv
          const [batch, inputHeight, inputWidth, inputChannels] = input.dims;
          let isDepthWiseConv = false;
          if (nGroups > 1) {
            if (nGroups !== inputChannels) {
              throw new Error('Group convolution is not supported.');
            } else {
              isDepthWiseConv = true;
              const nhwcData = convFilterTensor;
              // NHWC -> CHWN where C === 1
              const chwnData = new Float32Array(nhwcData.length);
              const N = convFilterDims[0];
              const H = convFilterDims[1];
              const W = convFilterDims[2];
              for (let n = 0; n < N; ++n) {
                for (let h = 0; h < H; ++h) {
                  for (let w = 0; w < W; ++w) {
                    chwnData[h*W*N + w*N + n] = nhwcData[n*H*W + h*W + w];
                  }
                }
              }

              convFilterTensor = chwnData;
              convFilterDims[0] = 1;
              convFilterDims[3] = nGroups;
            }
          }

          inputs.push(this._getNNTensorId(convNode.inputs[0]));
          inputs.push(this._addTensorFloat32(convFilterDims, convFilterTensor));
          inputs.push(this._addTensorFloat32([nChannels], convBiasTensor));

          inputs.push(this._addScalarInt32(paddingWidthBegin));
          inputs.push(this._addScalarInt32(paddingWidthEnd));
          inputs.push(this._addScalarInt32(paddingHeightBegin));
          inputs.push(this._addScalarInt32(paddingHeightEnd));
          inputs.push(this._addScalarInt32(strideX));
          inputs.push(this._addScalarInt32(strideY));

          if (isDepthWiseConv) {
            // set multiplier to 1, not used in onnx model
            inputs.push(this._addScalarInt32(1));
          }

          if (nextNode &&
              nextNode.opType === 'Relu' &&
              node.outputs[0] === nextNode.inputs[0]) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));
            i++;
            node = nextNode;
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

          // Add outputs
          const outputHeight = Math.floor((inputHeight-kernelHeight + paddingHeightBegin+paddingHeightEnd)/strideY + 1);
          const outputWidth = Math.floor((inputWidth - kernelWidth + paddingWidthBegin + paddingWidthEnd)/strideX + 1);
          const outputChannels = isDepthWiseConv ? nGroups : nChannels;
          const outputDims = [batch, outputHeight, outputWidth, outputChannels];
          const outputId = this._addTensorFloat32(outputDims);
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);

          opType = isDepthWiseConv ? this._nn.DEPTHWISE_CONV_2D : this._nn.CONV_2D;
        } break;
        case 'BatchNormalization': {
          // Add inputs
          const input = handler.getTensor(node.inputs[0]);
          const scale = handler.getTensor(node.inputs[1]);
          const bnBias = handler.getTensor(node.inputs[2]);
          const mean = handler.getTensor(node.inputs[3]);
          const variance = handler.getTensor(node.inputs[4]);
          const epsilon = attributes.getFloat('epsilon', 1e-5);

          const scaleTensor = scale.floatData;
          const meanTensor = mean.floatData;
          const varTensor = variance.floatData;
          const bnBiasTensor = bnBias.floatData;

          // Conv with identity kernel
          const nChannels = input.dims[3];
          const convFilterTensor = new Float32Array(nChannels * nChannels).fill(0);
          const convBiasTensor = new Float32Array(nChannels).fill(0);
          const convFilterDims = [nChannels, 1, 1, nChannels];
          const convBiasDims = [nChannels];

          for (let c = 0; c < nChannels; c++) {
            const w = scaleTensor[c] / Math.sqrt(varTensor[c] + epsilon);
            convFilterTensor[c * nChannels + c] = w;
            convBiasTensor[c] = bnBiasTensor[c] - w * meanTensor[c];
          }

          inputs.push(this._getNNTensorId(node.inputs[0]));
          inputs.push(this._addTensorFloat32(convFilterDims, convFilterTensor));
          inputs.push(this._addTensorFloat32(convBiasDims, convBiasTensor));
          // paddings
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          // strides
          inputs.push(this._addScalarInt32(1));
          inputs.push(this._addScalarInt32(1));

          let nextNode = this._graph.nodes[i + 1];
          if (nextNode &&
              nextNode.opType === 'Relu' &&
              node.outputs[0] === nextNode.inputs[0]) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));
            i++;
            node = nextNode;
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

          // Add outputs
          const outputDims = Array.from(input.dims);
          const outputId = this._addTensorFloat32(outputDims);
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);

          opType = this._nn.CONV_2D;
        } break;
        case 'Relu': {
          // Add inputs
          const input = handler.getTensor(node.inputs[0]);

          // Conv with identity kernel
          const nChannels = input.dims[3];
          const convFilterTensor = new Float32Array(nChannels * nChannels).fill(0);
          const convBiasTensor = new Float32Array(nChannels).fill(0);
          const convFilterDims = [nChannels, 1, 1, nChannels];
          const convBiasDims = [nChannels];

          for (let c = 0; c < nChannels; c++) {
            convFilterTensor[c * nChannels + c] = 1;
          }

          inputs.push(this._getNNTensorId(node.inputs[0]));
          inputs.push(this._addTensorFloat32(convFilterDims, convFilterTensor));
          inputs.push(this._addTensorFloat32(convBiasDims, convBiasTensor));
          // paddings
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          // strides
          inputs.push(this._addScalarInt32(1));
          inputs.push(this._addScalarInt32(1));
          inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));

          // Add outputs
          const outputDims = Array.from(input.dims);
          const outputId = this._addTensorFloat32(outputDims);
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);

          opType = this._nn.CONV_2D;
        } break;
        case 'Mul':
        case 'Sum':
        case 'Add': {

          if (node.opType === 'Sum' && node.inputs.length !== 2) {
            throw new Error(`Only support Sum with two inputs`);
          }
          const in1 = handler.getTensor(node.inputs[0]);
          const in2 = handler.getTensor(node.inputs[1]);
          inputs.push(this._getNNTensorId(node.inputs[0]));
          inputs.push(this._getNNTensorId(node.inputs[1]));

          if (node.opType === 'Add' || node.opType === 'Sum') {
            opType = this._nn.ADD;
          } else if (node.opType === 'Mul') {
            opType = this._nn.MUL;
          }

          let nextNode = this._graph.nodes[i + 1];
          if (nextNode &&
              nextNode.opType === 'Relu' &&
              node.outputs[0] === nextNode.inputs[0]) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));
            i++;
            node = nextNode;
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

          // Add outputs
          const in1Dims = in1.dims;
          const in2Dims = in2.dims;

          // Compatible dims (multidirectional broadcasting)
          const outputDims = new Array(Math.max(in1Dims.length, in2Dims.length));
          for (let i = in1Dims.length - 1, j = in2Dims.length - 1, k = outputDims.length - 1; k >= 0;) {
            let dim1 = in1Dims[i--] || 1;
            let dim2 = in2Dims[j--] || 1;
            if (dim1 !== dim2 && dim1 !== 1 && dim2 !== 1) {
              throw new Error(`Dimensions of ${in1} and ${in2} are not compatible`);
            }
            outputDims[k--] = Math.max(dim1, dim2);
          }

          const outputId = this._addTensorFloat32(outputDims);
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);
        } break;
        case 'Gemm': {
          // Add inputs
          const input = handler.getTensor(node.inputs[0]);    // A
          const weights = handler.getTensor(node.inputs[1]);  // B
          const bias = handler.getTensor(node.inputs[2]);     // C

          const alpha  = attributes.getFloat('alpha', 1.0);
          const beta   = attributes.getFloat('beta', 1.0);
          const transA = attributes.getInt('transA', 0);
          const transB = attributes.getInt('transB', 0);

          if (alpha !== 1 || beta !== 1 || transA || !transB) {
            throw new Error('Only support fc-like Gemm oprations, i.e. alpha == beta == 1 && !transA && transB');
          }

          const weightsId = this._addTensorFloat32(Array.from(weights.dims), weights.floatData as Float32Array);
          const biasId = this._addTensorFloat32(Array.from(bias.dims), bias.floatData as Float32Array);

          inputs.push(this._getNNTensorId(node.inputs[0]));
          inputs.push(weightsId);
          inputs.push(biasId);

          opType = this._nn.FULLY_CONNECTED;

          let nextNode = this._graph.nodes[i + 1];
          if (nextNode &&
              nextNode.opType === 'Relu' &&
              node.outputs[0] === nextNode.inputs[0]) {
            // Fuse relu
            inputs.push(this._addScalarInt32(this._nn.FUSED_RELU));
            i++;
            node = nextNode;
          } else {
            inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));
          }

          // Add outputs
          const nUnits = weights.dims[0];
          const batchSize = ShapeUtil.size(input.dims) / weights.dims[1];
          const outputDims = [batchSize, nUnits];
          const outputId = this._addTensorFloat32(outputDims);
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);
        } break;
        case 'AveragePool':
        case 'MaxPool': {
          const input = handler.getTensor(node.inputs[0]);
          inputs.push(this._getNNTensorId(node.inputs[0]));

          const pads = attributes.getInts('pads', [0, 0, 0, 0]);
          if (pads.length !== 4) {
            throw new Error('Invalid pads');
          }
          const paddingHeightBegin = pads[0];
          const paddingWidthBegin = pads[1];
          const paddingHeightEnd = pads[2];
          const paddingWidthEnd = pads[3];
          inputs.push(this._addScalarInt32(paddingWidthBegin));
          inputs.push(this._addScalarInt32(paddingWidthEnd));
          inputs.push(this._addScalarInt32(paddingHeightBegin));
          inputs.push(this._addScalarInt32(paddingHeightEnd));

          const strides = attributes.getInts('strides', [1, 1]);
          if (!strides || strides.length !== 2) {
            throw new Error('Invalid strides');
          }
          const strideY = strides[0];
          const strideX = strides[1];
          inputs.push(this._addScalarInt32(strideX));
          inputs.push(this._addScalarInt32(strideY));

          const kernelShape = attributes.getInts('kernel_shape', []);
          if (!kernelShape || kernelShape.length !== 2) {
            throw new Error('Invalid kernelShape');
          }
          const kernelHeight = kernelShape[0];
          const kernelWidth = kernelShape[1];
          inputs.push(this._addScalarInt32(kernelWidth));
          inputs.push(this._addScalarInt32(kernelHeight));
          inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

          const [batch, inputHeight, inputWidth, inputChannels] = input.dims;
          const outputHeight =
              Math.floor((inputHeight - kernelHeight + paddingHeightBegin + paddingHeightEnd)/strideY+1);
          const outputWidth = Math.floor((inputWidth - kernelWidth + paddingWidthBegin + paddingWidthEnd)/strideX + 1);
          const outputDims = [batch, outputHeight, outputWidth, inputChannels];
          const outputId = this._addTensorFloat32(outputDims);
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);

          if (node.opType === 'MaxPool') {
            opType = this._nn.MAX_POOL_2D;
          } else if (node.opType === 'AveragePool') {
            opType = this._nn.AVERAGE_POOL_2D;
          }
        } break;
        case 'Reshape': {
          const input = handler.getTensor(node.inputs[0]);
          const shape = handler.getTensor(node.inputs[1]);
          const shapeId = this._addTensorInt32(shape.integerData as Int32Array, Array.from(shape.dims));
          inputs.push(this._getNNTensorId(node.inputs[0]));
          inputs.push(shapeId);

          const inputDims = input.dims;
          let outputDims = Array.from(shape.integerData);
          // dim == 0 means actual dim is unchanged, i.e. taken from the inputDim
          outputDims = outputDims.map((d, i) => d === 0 ? inputDims[i] : d);
          // At most one dimension of the new shape can be -1
          const minusOneCnt = outputDims.filter(x => x === -1).length;
          if (minusOneCnt === 1) {
            const nonAdaptDim = outputDims.filter(x => x !== -1);
            const adaptDimIdx = outputDims.indexOf(-1);
            outputDims[adaptDimIdx] = ShapeUtil.size(inputDims) / ShapeUtil.size(nonAdaptDim);
          } else if (minusOneCnt !== 0) {
            throw new Error(`Invalid shape ${outputDims}`);
          }
          this._setOperandValue(shapeId, new Int32Array(outputDims));

          // Add outputs
          const outputId = this._addTensorFloat32(outputDims);
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);

          opType = this._nn.RESHAPE;
        } break;
        case 'Flatten': {
          const input = handler.getTensor(node.inputs[0]);
          const axis =  attributes.getInt('axis', 1);
          inputs.push(this._getNNTensorId(node.inputs[0]));

          const inputDims = input.dims;
          const rank = inputDims.length;
          if (axis > rank || axis < 0) {
            throw new Error(`Axis ${axis} is not in the range [0, ${rank}]`);
          }
          let outputDim1 = inputDims.slice(0, axis);
          let outputDim2 = inputDims.slice(axis);
          const outputDims = [
            outputDim1.length ? ShapeUtil.size(outputDim1) : 1,
            outputDim2.length ? ShapeUtil.size(outputDim2) : 1
          ];

          const shapeId = this._addTensorInt32(new Int32Array(outputDims), [2]);
          inputs.push(shapeId);

          // Add output
          const outputId = this._addTensorFloat32(outputDims);
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);

          opType = this._nn.RESHAPE;
        } break;
        case 'Concat': {
          for (let i = 0; i < node.inputs.length; ++i) {
            inputs.push(this._getNNTensorId(node.inputs[i]));
          }

          const axis = attributes.getInt('axis');
          if (axis && axis !== 1) {
            throw new Error(`Invalid axis ${axis}`);
          }
          // C axis is 3 in NHWC layout
          const concatAxis = 3;
          inputs.push(this._addScalarInt32(concatAxis));

          // Add output
          let outputDims = Array.from(handler.getTensor(node.inputs[0]).dims);
          for (let i = 1; i < node.inputs.length; ++i) {
            outputDims[concatAxis] += handler.getTensor(node.inputs[i]).dims[concatAxis];
          }
          const outputId = this._addTensorFloat32(outputDims);
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);

          opType = this._nn.CONCATENATION;
        } break;
        case 'Dropout': {
          // bypass Dropout node
          const input = handler.getTensor(node.inputs[0]);
          handler.setTensor(node.outputs[0], input);
          continue;
        };
        case 'GlobalAveragePool': {
          const input = handler.getTensor(node.inputs[0]);
          inputs.push(this._getNNTensorId(node.inputs[0]));
          // paddings
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          inputs.push(this._addScalarInt32(0));
          // strides
          inputs.push(this._addScalarInt32(1));
          inputs.push(this._addScalarInt32(1));
          // filters
          const [batch, inputHeight, inputWidth, inputChannels] = input.dims;
          inputs.push(this._addScalarInt32(inputWidth));
          inputs.push(this._addScalarInt32(inputHeight));
          inputs.push(this._addScalarInt32(this._nn.FUSED_NONE));

          // Add outputs
          const outputHeight = 1;
          const outputWidth = 1;
          const outputDims = [batch, outputHeight, outputWidth, inputChannels];
          const outputId = this._addTensorFloat32(outputDims);
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);

          opType = this._nn.AVERAGE_POOL_2D;
        } break;
        case 'Softmax': {
          const input = handler.getTensor(node.inputs[0]);
          inputs.push(this._getNNTensorId(node.inputs[0]));
          // Set beta to 1.0
          inputs.push(this._addScalarFloat32(1.0));

          const outputDims = input.dims;
          const outputId = this._addTensorFloat32(Array.from(outputDims));
          const outputTensor = new Tensor(outputDims, 'float32');
          this._setOnnxIdToNNId(node.outputs[0], outputId);
          handler.setTensor(node.outputs[0], outputTensor);
          outputs.push(outputId);

          opType = this._nn.SOFTMAX;
        } break;
        default: {
          throw new Error(`${node.opType} is not supported.}`);
        }
      }

      this._addOperation(opType, inputs, outputs);
    }
  }

  _setOperandValue(index: number, value: NNTensorType) {
    this._model.setOperandValue(index, value);
  }

  _addOperand(type: OperandOptions, value?: NNTensorType) {
    let index = this._operandIndex++;
    this._model.addOperand(type);
    if (typeof value !== 'undefined') {
      this._setOperandValue(index, value);
    }
    return index;
  }

  _addOperation(opCode: number, inputs: number[], outputs: number[]) {
    this._model.addOperation(opCode, inputs, outputs);
  }

  _addScalarInt32(value: number) {
    return this._addOperand({type: this._nn.INT32}, new Int32Array([value]));
  }

  _addScalarFloat32(value: number) {
    return this._addOperand({type: this._nn.FLOAT32}, new Float32Array([value]));
  }

  _addTensorFloat32(dims: number[], tensor?: Float32Array) {
    return this._addOperand({
      type: this._nn.TENSOR_FLOAT32,
      dimensions: dims
    }, tensor);
  }

  _addTensorInt32(tensor: Int32Array, dims: number[]) {
    return this._addOperand({
      type: this._nn.TENSOR_INT32,
      dimensions: dims
    }, tensor);
  }

  private _nn: NeuralNetworkContext;
  private _graph: WebNNGraphOp;
  private _operandIndex: number;
  private _onnxIdToNNId: Map<number, number>;
  private _model: Model;
  private _compilation: Compilation;
  private _execution: Execution;
}
