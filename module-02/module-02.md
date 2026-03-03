# module-02

The script trains the model on the small sample dataset in `index.js` and prints prediction probabilities for the example person included at the end of the file.

## Goal

This module contains a minimal TensorFlow.js example for Node.js showing a complete train-and-predict flow.

- **Library used:** `@tensorflow/tfjs-node` (native TensorFlow bindings for Node).
- **Model type:** `tf.sequential()` with two dense layers:
    - Hidden layer: input shape `[7]`, 80 units, `relu` activation.
    - Output layer: 3 units, `softmax` activation (multi-class classification).
- **Compilation:** optimizer `adam`, loss `categoricalCrossentropy`, metric `accuracy`.
- **Training data format:** 2D tensors where each input row is 7 normalized features: `[age_norm, blue, red, green, SP, Rio, Curitiba]` and outputs are one-hot vectors for classes `[premium, medium, basic]`.
- **Training:** `model.fit` runs for 100 epochs with shuffling; per-epoch loss is logged to the console.
- **Prediction:** Convert a new sample to a 2D tensor and call `model.predict(...)`; the script prints the predicted class probabilities.

## Quick run

1. Install dependencies:

```
npm install
```

2. Start the example:

```
npm start
```


