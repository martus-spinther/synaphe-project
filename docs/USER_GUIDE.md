# Synaphe User Guide

## What is Synaphe?

Synaphe is a programming language designed for people building hybrid AI and quantum computing applications. It transpiles to Python, so you get access to all existing libraries (PyTorch, Qiskit, PennyLane, NumPy), while Synaphe adds compile-time safety that Python lacks.

If you've ever hit a `RuntimeError: size mismatch` in PyTorch at 3am, or spent hours debugging why your quantum circuit produces garbage data — Synaphe was built for you.

## Who is this for?

- **ML researchers** who want tensor shape errors caught before training starts
- **Quantum computing researchers** who want the No-Cloning Theorem enforced by the compiler
- **Data scientists** who want clean pipeline syntax for AI workflows
- **Students** learning quantum ML who want readable, auditable code

## Core Concepts

### 1. The Pipeline Operator `|>`

This is Synaphe's signature feature. Instead of nested function calls, you chain operations left to right:

```
// Python: softmax(model.forward(reshape(normalize(data), (1, 784))))
// Synaphe:
let prediction = data |> normalize |> reshape(1, 784) |> model.forward |> softmax
```

Every golden snippet uses this. It makes code read like the flow of data, not a puzzle of parentheses.

### 2. Typed Tensors

```
let x: Tensor<Float32, [32, 784]> = randn(32, 784)
let w: Tensor<Float32, [784, 128]> = randn(784, 128)
let y = x @ w   // Compiler checks: [32, 784] @ [784, 128] → [32, 128] ✓

let bad = x @ x  // ✗ Compile error: [32, 784] @ [32, 784] — inner dims don't match
```

Symbolic dimensions work too: `Tensor<Float32, [batch, 784]>` lets the batch size vary while the compiler still checks the 784.

### 3. Quantum Types Are Linear

In quantum mechanics, you cannot copy a quantum state (No-Cloning Theorem). Synaphe enforces this:

```
let q = qregister(4)        // Fresh quantum register
let result = measure(q)      // OK — q is consumed, result is classical
let oops = hadamard(q)       // ✗ Error: q already measured!
```

The type system tracks four states: FRESH → ACTIVE → MEASURED → CONSUMED. Once a qubit is measured or passed to another function, it's gone.

But measurement results ARE classical — you can copy them freely:

```
let bits = measure(q)        // Classical result
let a = bits                 // OK — classical, can copy
let b = bits                 // OK — still classical
```

### 4. Models Are Declarative

```
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
```

This transpiles to a complete `nn.Module` subclass with `__init__` and `forward` methods.

### 5. Schemas Validate Data

```
schema SensorReading {
    temperature: Float where temperature >= -273.15
    pressure: Float where pressure >= 0.0
    device_id: String
}
```

Data is validated when it crosses a boundary (loaded from file, received from API). Invalid data never reaches your model.

### 6. Automatic Differentiation

For classical functions, `grad()` uses backpropagation. For quantum circuits, it automatically applies the Parameter Shift Rule:

```
@differentiable
fn quantum_cost(theta: Float) -> Float {
    qubit() |> ry(theta) |> measure |> expectation(PauliZ)
}

let gradient = grad(quantum_cost)
let dtheta = gradient(0.5)
// Uses: df/dθ = [f(θ+π/2) - f(θ-π/2)] / 2
```

You never write the Parameter Shift Rule yourself. The compiler handles it.

### 7. Hardware-Aware Checking

Before submitting to a QPU, Synaphe checks your circuit against real hardware specs:

```
check_fidelity(my_circuit, target="ibm_brisbane")
// ✗ [coherence] Circuit duration exceeds T2 time — results will be noise
// ⚠ [fidelity] Estimated fidelity 23% — consider error mitigation
// ⚠ [connectivity] CNOT(0,15) requires SWAP insertion
```

Built-in profiles for IBM Brisbane, IBM Sherbrooke, Google Sycamore, and IonQ Forte.

## Language Reference

### Variables

```
let x = 42              // Immutable by default
let name = "Alice"
let flag = true
let pi = 3.14159
```

### Functions

```
fn add(a: Int, b: Int) -> Int {
    return a + b
}

fn greet(name: String) -> String {
    return "Hello, " + name
}
```

### Control Flow

```
if score > 0.9 {
    print("Excellent")
} else if score > 0.7 {
    print("Good")
} else {
    print("Needs work")
}

for i in range(10) {
    print(i)
}

while condition {
    // loop body
}
```

### Pattern Matching

```
match result {
    0 => print("zero"),
    1 => print("one"),
    n => print(n)           // Binds n to the value
}
```

### Imports

```
import torch                          // Import a Python module
from torch import nn, optim           // Import specific names
import numpy as np                    // With alias
```

### Matrix Multiplication

```
let y = x @ w       // Uses @ operator, just like Python/NumPy
```

### Comments

```
// Single-line comment
/* Multi-line
   comment */
```

## Running Synaphe

### REPL (Interactive Mode)

```bash
python synaphe_cli.py
```

This launches the interactive REPL where you can type Synaphe code and see it compiled to Python immediately.

REPL commands:
- `.tokens` — show the token stream for the last input
- `.ast` — show the abstract syntax tree
- `.python` — show the full transpiled Python output
- `.run` — compile and execute the last input
- `.example pipeline` — load a built-in example

### Compile a File

```bash
python synaphe_cli.py build examples/mnist.flux
```

### Compile and Run

```bash
python synaphe_cli.py run examples/mnist.flux
```

### Type Check Only

```bash
python synaphe_cli.py check examples/mnist.flux
```

## Standard Library

### Quantum Operations

| Function | Description |
|----------|-------------|
| `qubit()` | Allocate a single qubit in \|0⟩ |
| `qregister(n)` | Allocate n qubits in \|00...0⟩ |
| `hadamard(q)` | Apply Hadamard gate |
| `rx(theta, q)` | X-rotation by theta |
| `ry(theta, q)` | Y-rotation by theta |
| `rz(theta, q)` | Z-rotation by theta |
| `cx(q)` | Controlled-NOT (CNOT) |
| `measure(q)` | Measure all qubits → classical bits |
| `expectation(op, q)` | Compute ⟨ψ\|O\|ψ⟩ |

### Optimization

| Function | Description |
|----------|-------------|
| `minimize(cost_fn, init, optimizer)` | Minimize a cost function |
| `grad(fn)` | Compute gradient (classical or quantum) |
| `GradientDescent(lr, steps)` | Gradient descent optimizer |
| `Adam(lr, steps)` | Adam optimizer |
| `COBYLA(maxiter)` | Constraint optimization |

### Chemistry

| Function | Description |
|----------|-------------|
| `hamiltonian(molecule, basis, bondlength)` | Build molecular Hamiltonian |
| `prepare_hartree_fock(q, occupation)` | Prepare Hartree-Fock initial state |
| `double_excitation(theta, q)` | Double excitation gate for VQE |

### Data Pipeline

| Function | Description |
|----------|-------------|
| `load_csv(path, schema)` | Load CSV with optional validation |
| `generate_synthetic(n, features, classes)` | Generate test data |
| `normalize(data)` | Normalize to mean=0, std=1 |
| `batch(data, size)` | Split into batches |
| `shuffle(data, seed)` | Random shuffle |
| `one_hot(labels, n_classes)` | One-hot encode labels |
| `split(dataset, ratio)` | Train/test split |

## FAQ

**Is Synaphe a new Python framework?**
No — it's a new language that compiles TO Python. You write Synaphe code, the compiler checks it for type safety, and outputs clean Python that uses PyTorch, PennyLane, etc.

**Do I need a quantum computer?**
No. Synaphe includes a built-in quantum simulator. The same code runs on the simulator during development and on real hardware when you're ready.

**Can I use my existing Python libraries?**
Yes. `import torch`, `import pandas`, `import sklearn` — all work directly in Synaphe.

**How is this different from Qiskit?**
Qiskit is a quantum SDK (library). Synaphe is a programming language that can USE Qiskit as a backend. Synaphe adds compile-time type safety, the pipeline operator, and automatic differentiation across classical-quantum boundaries — things Qiskit doesn't provide.

**Is the quantum simulation accurate?**
The built-in simulator uses exact statevector simulation with NumPy. It's mathematically exact (no noise) for circuits up to ~20 qubits. For noisy simulation, use the PennyLane or Qiskit backends.

## Next Steps

1. Run `python examples/demo.py` to see all features in action
2. Try the REPL: `python synaphe_cli.py`
3. Read the golden snippets: `examples/golden_snippets.flux`
4. Check out the test suite to understand edge cases: `tests/`
5. Join the community and contribute!
