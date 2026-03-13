// ================================================================
// FLUX Example: MNIST Classifier with Pipeline
// A practical ML workflow in clean, type-safe syntax
// ================================================================

// Import Python libraries seamlessly
import torch
from torch import nn

// Define data validation schema
schema MNISTSample {
    pixels: Tensor<Float32, [28, 28]>
    label: Int where label >= 0
}

// Define the model declaratively
model MNISTClassifier {
    layers: [
        Linear(784, 256),
        ReLU,
        Dropout(0.3),
        Linear(256, 128),
        ReLU,
        Linear(128, 10)
    ]
    loss: CrossEntropy
    optimizer: Adam(lr=0.001)
}

// Training function with hardware target
fn train(model: MNISTClassifier, epochs: Int = 10) -> Float {
    let best_acc = 0.0
    for epoch in range(epochs) {
        let loss = model.train_epoch()
        let acc = model.evaluate()
        if acc > best_acc {
            best_acc = acc
        }
        print("Epoch {epoch}: loss={loss}, acc={acc}")
    }
    return best_acc
}

// Clean pipeline for inference
fn predict(model: MNISTClassifier, image: Tensor<Float32, [28, 28]>) -> Int {
    let result = image
        |> reshape(1, 784)
        |> model.forward
        |> softmax
    return result.argmax()
}
