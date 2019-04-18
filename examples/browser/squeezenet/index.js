async function runExample() {

  printHeader('Results');

  const supportedOps = getSupportedOps();
  log(`Offload Ops: ${supportedOps.length ? supportedOps.join(', ') : 'None'}`);
  log(`Enable Pseudo Reorder: ${pseudoReorder.checked}`);
  log(`Enable Op-level profiling: ${profiling.checked}`);
  log(`Iterations: ${Number(iteration.value)}`);

  // Create an ONNX inference session with WebGL backend.
  const session = new onnx.InferenceSession({
    backendHint: 'wasm',
    supportedOps: supportedOps,
    enablePseudoReorder: pseudoReorder.checked
  });

  // Load an ONNX model. This model is SqueezeNet that takes a 1*3*224*224 image and classifies it.
  await session.loadModel("../../../deps/data/data/examples/models/squeezenetV1_8.onnx");

  // load image.
  const imageLoader = new ImageLoader(imageSize, imageSize);
  const imageData = await imageLoader.getImageData('./squeezenet-piano.jpg');

  // preprocess the image data to match input dimension requirement, which is 1*3*224*224
  const width = imageSize;
  const height = imageSize;
  const preprocessedData = preprocess(imageData.data, width, height);

  const inputTensor = new onnx.Tensor(preprocessedData, 'float32', [1, 3, width, height]);
  // Run model with Tensor inputs and get the result.
  const outputMap = await session.run([inputTensor]);
  const outputData = outputMap.values().next().value.data;

  // Render the output result in html.
  printMatches(outputData);

  runBenchmark(session, inputTensor, Number(iteration.value), profiling.checked);
}

/**
 * Preprocess raw image data to match SqueezeNet requirement.
 */
function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);

  // Normalize 0-255 to (-1)-1
  ndarray.ops.subseq(dataFromImage.pick(2, null, null), 103.939);
  ndarray.ops.subseq(dataFromImage.pick(1, null, null), 116.779);
  ndarray.ops.subseq(dataFromImage.pick(0, null, null), 123.68);

  // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
  ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
  ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));

  return dataProcessed.data;
}

/**
 * Utility function to post-process SqueezeNet output. Find top k ImageNet classes with highest probability.
 */
function imagenetClassesTopK(classProbabilities, k) {
  if (!k) { k = 5; }
  const probs = Array.from(classProbabilities);
  const probsIndices = probs.map(
    function (prob, index) {
      return [prob, index];
    }
  );
  const sorted = probsIndices.sort(
    function (a,b) {
      if(a[0] < b[0]) {
        return -1;
      }
      if(a[0] > b[0]) {
        return 1;
      }
      return 0;
    }
  ).reverse();
  const topK = sorted.slice(0, k).map(function (probIndex) {
    const iClass = imagenetClasses[probIndex[1]];
    return {
      id: iClass[0],
      index: parseInt(probIndex[1], 10),
      name: iClass[1].replace(/_/g, ' '),
      probability: probIndex[0]
    };
  });
  return topK;
}

/**
 * Render SqueezeNet output to Html.
 */
function printMatches(data) {
  let outputClasses = [];
  if (!data || data.length === 0) {
    const empty = [];
    for (let i = 0; i < 5; i++) {
      empty.push({ name: '-', probability: 0, index: 0 });
    }
    outputClasses = empty;
  } else {
    outputClasses = imagenetClassesTopK(data, 5);
  }
  const results = [];
  for (let i of [0, 1, 2, 3, 4]) {
    results.push(`${outputClasses[i].name}: ${Math.round(100 * outputClasses[i].probability)}%`);
  }
  log();
  log(results.join('<br/>'));
}
