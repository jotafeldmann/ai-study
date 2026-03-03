import tf from '@tensorflow/tfjs-node'; // TensorFlow for Node.js

// Function to train the neural network
async function trainModel(inputXs, outputYs) {

    const model = tf.sequential(); // Create sequential model

    // Hidden layer
    model.add(
        tf.layers.dense({
            inputShape: [7],      // 7 input features
            units: 80,            // 80 neurons
            activation: 'relu'    // Activation function
        })
    );

    // Output layer
    model.add(
        tf.layers.dense({
            units: 3,             // 3 output classes
            activation: 'softmax' // Multi-class classification
        })
    );

    model.compile({
        optimizer: 'adam',               // Optimization algorithm
        loss: 'categoricalCrossentropy', // Loss for multi-class
        metrics: ['accuracy']            // Track accuracy
    });

    await model.fit(inputXs, outputYs, {
        epochs: 100,        // Training iterations
        shuffle: true,      // Shuffle data
        verbose: 0,         // Hide default logs
        callbacks: {
            onEpochEnd: (epoch, logs) =>  // Custom log
                console.log(
                    `Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}`
                )
        }
    });

    return model; // Return trained model
}

// Function to make predictions
async function predict(model, person) {

    const tfInput = tf.tensor2d(person); // Convert to tensor

    const prediction = model.predict(tfInput); // Run model

    const predictedValues = await prediction.array(); // Convert to array

    console.log(predictedValues); // Show probabilities
}

// Normalized training data
// [age_norm, blue, red, green, SP, Rio, Curitiba]
const normalizedPeopleTensor = [
    [0.33, 1, 0, 0, 1, 0, 0], // Person 1
    [0, 0, 1, 0, 0, 1, 0],    // Person 2
    [1, 0, 0, 1, 0, 0, 1]     // Person 3
];

// Class labels (one-hot)
// [premium, medium, basic]
const tensorLabels = [
    [1, 0, 0], // premium
    [0, 1, 0], // medium
    [0, 0, 1]  // basic
];

const inputXs = tf.tensor2d(normalizedPeopleTensor); // Input tensor
const outputYs = tf.tensor2d(tensorLabels);          // Output tensor

const model = await trainModel(inputXs, outputYs); // Train model

// New person to classify
const normalizedPersonTensor = [
    [
        0.2, // normalized age
        0,   // blue
        0,   // red
        1,   // green
        0,   // SP
        0,   // Rio
        1    // Curitiba
    ]
];

await predict(model, normalizedPersonTensor); // Predict class
