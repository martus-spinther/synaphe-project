# Synaphe — Language Design Specification v0.1

## The Problem (Verified, Real-World)

### Primary Problem: The Hybrid AI-Quantum Workflow Gap

The AI/ML ecosystem and the quantum computing ecosystem are developing in parallel
but speak completely different languages. This creates **five verified pain points**:

1. **Tensor Shape Errors Are the #1 Runtime Bug in ML**
   PyTorch's most common runtime error is `RuntimeError: size mismatch`. These are
   caught only at runtime, often deep in training loops. No existing language catches
   tensor dimension mismatches at compile time.

2. **Quantum-Classical Handoff Is Manual and Fragile**
   Hybrid quantum-classical workflows (VQE, QAOA, quantum kernel methods) require
   manually shuttling data between Qiskit/Cirq/PennyLane and PyTorch/JAX. There is
   no unified type system spanning classical tensors and quantum states.

3. **Pipeline Fragmentation**
   A typical ML workflow requires Python + YAML configs + shell scripts + framework
   DSLs + deployment configs. MLOps tooling (30+ competing tools) tries to paper
   over this but adds complexity.

4. **No Compile-Time Safety for AI Workflows**
   Python's dynamic typing means model architecture errors, data pipeline mismatches,
   and device placement bugs (CPU vs GPU vs QPU) are found only at runtime.

5. **Quantum Frameworks Are Siloed**
   Qiskit targets IBM hardware, Cirq targets Google, PennyLane bridges them but adds
   another abstraction layer. Code written for one backend rarely ports cleanly.

### Why Now?

- Quantum-specific ML hardware entering market (2026)
- Hybrid quantum-classical workflows moving from research to production
- Python tooling renaissance (uv, ruff) proves demand for better DX
- 70%+ of quantum computing jobs require Python — the ecosystem is ready for a layer above it

## Design Philosophy

```
"Catch at compile time what today crashes at runtime.
Make quantum feel as natural as GPU.
Transpile to Python so nothing breaks."
```

### Core Principles

1. **Practical First** — Every feature solves a documented, real-world problem
2. **Python Interop** — Transpiles to Python; import any Python library directly
3. **Gradual Typing** — Start dynamic, add types as you scale
4. **Hardware Agnostic** — CPU, GPU, QPU are deployment targets, not code changes
5. **Pipeline Native** — Data pipelines are a first-class language construct
6. **Quantum Ready** — Quantum types and operations built into the type system

## Language Overview

### Type System

```flux
// Tensor types with compile-time shape checking
let x: Tensor<Float32, [batch, 784]> = load("mnist.csv")
let w: Tensor<Float32, [784, 128]> = init.xavier([784, 128])
let y = x @ w  // Compiler verifies: [batch, 784] @ [784, 128] -> [batch, 128]

// Quantum types
let q: Qubit = qubit()
let reg: QRegister<4> = qregister(4)

// Probabilistic types
let p: Prob<Float32> = measure(q)  // Returns probability distribution
let sample: Float32 = p.sample()

// Union types for hybrid results
type HybridResult = Classical(Tensor) | Quantum(QState) | Error(String)
```

### Pipeline Operator

```flux
// The |> operator chains operations — the core workflow primitive
let predictions = data
    |> validate(schema)
    |> normalize(mean=0.0, std=1.0)
    |> batch(size=32)
    |> model.forward
    |> softmax

// Quantum pipeline
let result = qregister(4)
    |> hadamard(all)
    |> cx(0, 1)
    |> rz(theta, 2)
    |> measure
```

### Hardware Targets

```flux
// Declare execution target — compiler handles the rest
@target(gpu)
fn train(model: Model, data: DataLoader) -> Metrics {
    ...
}

@target(qpu: "ibm_brisbane")
fn quantum_kernel(x: Tensor<Float32, [n, d]>) -> Tensor<Float32, [n, n]> {
    ...
}

// Or let the compiler choose optimally
@target(auto)
fn hybrid_vqe(hamiltonian: Operator) -> Float64 {
    ...
}
```

### Model Definition

```flux
// Clean, declarative model definition
model Classifier {
    layers: [
        Linear(784, 256),
        ReLU,
        Dropout(0.3),
        Linear(256, 10)
    ]
    
    loss: CrossEntropy
    optimizer: Adam(lr=0.001)
}

// Quantum-enhanced model
model QuantumClassifier {
    layers: [
        Linear(784, 16),
        QuantumLayer(qubits=4, depth=3),  // <-- Quantum circuit as a layer
        Linear(4, 10)
    ]
    
    loss: CrossEntropy
    optimizer: Adam(lr=0.001)
}
```

### Data Validation (Inspired by Elo)

```flux
// Schemas are first-class types — validated at boundaries
schema MNISTSample {
    pixels: Tensor<Float32, [28, 28]> where all(0.0 <= _ <= 1.0)
    label: Int where 0 <= _ <= 9
}

// Compile-time guarantee: if it type-checks, the data is valid
fn train(data: Dataset<MNISTSample>) -> Model {
    ...
}
```

### Pattern Matching

```flux
// Exhaustive pattern matching for model results
match result {
    Classical(tensor) => print("Got tensor: {tensor.shape}")
    Quantum(state) => print("Got quantum state: {state.n_qubits} qubits")
    Error(msg) => log.error(msg)
}

// Pattern match on tensor shapes
match tensor.shape {
    [1, n] => squeeze(0)       // Remove batch dim if batch=1
    [b, n] => pass             // Normal batched tensor
    _ => error("Unexpected shape: {tensor.shape}")
}
```

### Automatic Differentiation

```flux
// Gradients are a language-level concept
fn loss_fn(params: Tensor, data: Tensor) -> Float32 {
    let pred = model(params, data)
    return mse(pred, data.labels)
}

// grad() is a built-in, not a library call
let gradients = grad(loss_fn, wrt=params)
let updated = params - lr * gradients

// Works on quantum circuits too
@differentiable
fn quantum_circuit(theta: Float32) -> Float32 {
    qubit()
        |> rx(theta)
        |> measure
        |> expectation(PauliZ)
}

let dtheta = grad(quantum_circuit, wrt=theta)
```

## Compilation Targets

### Phase 1 (Now): Python Transpilation
- Output idiomatic Python + PyTorch/JAX
- Quantum code outputs Qiskit/PennyLane
- Full interop: `import torch`, `import qiskit` just work

### Phase 2 (6 months): MLIR Backend
- Compile to MLIR for optimized CPU/GPU execution
- Enable whole-program optimization

### Phase 3 (12 months): QIR Backend
- Compile quantum portions to QIR (Quantum Intermediate Representation)
- Target any QIR-compatible quantum hardware

## File Extension

`.synaphe`

## Standard Library Modules

- `synaphe.tensor` — Tensor operations with shape types
- `synaphe.quantum` — Quantum gates, circuits, measurement
- `synaphe.data` — Data loading, validation, pipelines
- `synaphe.optim` — Optimizers with type-safe hyperparameters
- `synaphe.metrics` — Evaluation metrics
- `synaphe.viz` — Visualization utilities
- `synaphe.io` — File I/O with format detection
