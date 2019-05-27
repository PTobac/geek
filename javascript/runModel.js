
async function runModel(session, ctx) {

    // load image.
    const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
    const { data, width, height } = imageData;
    console.log('orig ctx: '+JSON.stringify(data));
    console.log('orig width: '+width);
    console.log('orig height: '+height);
    // preprocess the image data to match input dimension requirement, which is 1*3*224*224
    const preprocessedData = sqnPreprocess(data, width, height);
    const inputTensor = new onnx.Tensor(preprocessedData, 'float32', [1, 3, width, height]);
    // Run model with Tensor inputs and get the result.
    console.log('preprocessed tensor: '+JSON.stringify(inputTensor));
    const outputMap  = await session.run([inputTensor]);
    const outputData = outputMap.values().next().value.data;
    console.log(outputData);
    return (outputData);

}
  /**
   * Preprocess raw image data to match SqueezeNet requirement.
   */
  function sqnPreprocess(data, width, height) {
    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);

    // Normalize 0-255 to (-1)-1
    ndarray.ops.subseq(dataFromImage.pick(null, null,2), 85.0);
    ndarray.ops.subseq(dataFromImage.pick(null, null,1), 111.0);
    ndarray.ops.subseq(dataFromImage.pick(null, null,0), 139.0);

    // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
    ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
    ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
    ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));

    return dataProcessed.data;
  }

