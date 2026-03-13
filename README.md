<img src="https://raw.githubusercontent.com/martus-spinther/synaphe-project/main/banner.svg" alt="Synaphe — The bridge between AI and quantum computing" width="100%">
<p align="center">
  <h1 align="center">Synaphe</h1>
  <p align="center">
    <em>A programming language for hybrid AI and quantum computing</em>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#what-synaphe-does">What It Does</a> •
    <a href="#golden-snippets">Golden Snippets</a> •
    <a href="#installation">Installation</a> •
    <a href="#documentation">Docs</a> •
    <a href="CONTRIBUTING.md">Contributing</a>
  </p>
</p>

---

**Synaphe** (from the Greek *synaphe*, meaning "connection" or "junction") is a programming language that bridges classical AI and quantum computing. It transpiles to Python, giving you the entire PyTorch and Qiskit/PennyLane ecosystem — while adding three things Python lacks:

1. **Tensor shape checking at compile time** — catches the #1 PyTorch runtime error before your code runs
2. **Linear quantum types** — enforces the No-Cloning Theorem through the type system, preventing qubit reuse bugs
3. **Native automatic differentiation across quantum circuits** — `grad()` uses the Parameter Shift Rule for quantum gates and backpropagation for classical tensors, seamlessly

## What problem does this solve?

Today, building a hybrid quantum-classical ML workflow requires juggling Python, PyTorch, Qiskit, PennyLane, YAML configs, and dozens of MLOps tools. Tensor shape mismatches crash at runtime. Qubit reuse bugs produce garbage data silently. And computing gradients through quantum circuits requires manual Parameter Shift Rule implementations.

**Synaphe reduces a 65-line VQE implementation to 18 lines** — and catches shape errors, qubit bugs, and hardware constraint violations before your code ever touches a QPU.

## Quick Start

```bash
# Clone the repository
git clone https://github.com/martus-spinther/synaphe-project.git
cd synaphe

# Run the interactive demo
python examples/demo.py

# Run the test suite (86 tests)
python tests/test_parser.py
python tests/test_typechecker.py
python tests/test_v030.py
```

### Requirements

- Python 3.9+
- NumPy (for quantum simulation)
- Optional: PyTorch, PennyLane, Qiskit (for hardware execution)

## Golden Snippets

These side-by-side comparisons show why Synaphe exists:

### VQE — Finding the ground state of H₂

<table>
<tr><th>Python / PennyLane (65 lines)</th><th>Synaphe (18 lines)</th></tr>
<tr><td>

```python
import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem

symbols = ["H", "H"]
coordinates = np.array([0.0, 0.0, -0.6614,
                        0.0, 0.0,  0.6614])
H, qubits = qchem.molecular_hamiltonian(
    symbols, coordinates)
dev = qml.device("default.qubit", wires=qubits)
hf_state = qchem.hf_state(2, qubits)

@qml.qnode(dev)
def circuit(param):
    qml.BasisState(hf_state, wires=range(qubits))
    qml.DoubleExcitation(param, wires=[0, 1, 2, 3])
    return qml.expval(H)

opt = qml.GradientDescentOptimizer(stepsize=0.4)
theta = np.array(0.0, requires_grad=True)
for n in range(100):
    theta, energy = opt.step_and_cost(circuit, theta)
# + device setup, HF encoding, QNode decoration...
```

</td><td>

```
let H = hamiltonian("H2", basis="sto-3g")

@differentiable
fn ansatz(theta: Float, q: QRegister<4>) -> QState<4> {
    q
        |> prepare_hartree_fock([1, 1, 0, 0])
        |> double_excitation(theta)
}

let ground_state = minimize(
    fn(theta) => expectation(H, ansatz(theta, qregister(4))),
    init = 0.0,
    optimizer = GradientDescent(lr=0.4, steps=100)
)

print("Energy: {ground_state.energy} Ha")
// No device. No QNode. No HF encoding. Just math.
```

</td></tr>
</table>

See [`examples/golden_snippets.synaphe`](examples/golden_snippets.synaphe) for 5 complete comparisons (VQE, hybrid ML, QAOA, differentiable chemistry, anomaly detection).

## Architecture

```
synaphe/
├── src/                    # Language implementation
│   ├── lexer.py           # Tokenizer (40+ token types)
│   ├── ast_nodes.py       # Abstract syntax tree
│   ├── parser.py          # Recursive descent parser
│   ├── types.py           # Type system (tensor, quantum, differentiable)
│   ├── typechecker.py     # Three-pillar type checker
│   ├── transpiler.py      # Python code generator
│   └── hardware.py        # Hardware-aware constraint checking
├── stdlib/                 # Standard library
│   ├── runtime.py         # Quantum simulation, autodiff, optimization
│   └── data.py            # Data loading, validation, transforms
├── examples/              # Example programs
│   ├── demo.py            # Executable demo of all features
│   ├── golden_snippets.synaphe
│   ├── mnist.flux
│   └── quantum_hybrid.flux
├── tests/                 # Test suite (86 tests)
│   ├── test_parser.py     # 47 lexer/parser tests
│   ├── test_typechecker.py # 19 type checker tests
│   └── test_v030.py       # 20 hardware/data tests
└── docs/                  # Documentation
    ├── DESIGN.md          # Language specification
    ├── DESIGN_v2.md       # Revised design principles
    └── USER_GUIDE.md      # Getting started guide
```

## The Three Pillars

### Pillar 1: Tensor Shape Safety

```
let x: Tensor<Float32, [32, 784]> = randn(32, 784)
let w: Tensor<Float32, [10, 128]> = randn(10, 128)
let y = x @ w
// ✗ ShapeMismatchError: inner dimensions 784 vs 10 do not match
```

### Pillar 2: Linear Quantum Types

```
let q = qregister(4)
let result = measure(q)    // OK — consumes q
let oops = hadamard(q)     // ✗ LinearityError: quantum state already measured
```

### Pillar 3: Autodiff Bridge

```
@differentiable
fn circuit(theta: Float) -> Float {
    qubit() |> ry(theta) |> measure |> expectation(PauliZ)
}

let gradient = grad(circuit)  // Uses Parameter Shift Rule automatically
```

### Hardware-Aware Checking

```
// check_fidelity warns before you waste QPU time:
// ✗ [coherence] Circuit duration (17.6 μs) exceeds T2 (10 μs)
// ⚠ [fidelity] Estimated fidelity 12.3% — results dominated by noise
// ⚠ [connectivity] CNOT(0, 15) — qubits not directly connected
```

Includes real profiles for IBM Brisbane, IBM Sherbrooke, Google Sycamore, and IonQ Forte.

## Roadmap

- [x] **v0.1** — Lexer, parser, Python transpiler, REPL
- [x] **v0.2** — Type system, linear quantum types, autodiff bridge
- [x] **v0.3** — Hardware constraint checking, data pipelines
- [ ] **v0.4** — End-to-end `.synaphe` file compilation and execution
- [ ] **v0.5** — Language server (LSP) for IDE support
- [ ] **v0.6** — MLIR backend for optimized classical execution
- [ ] **v0.7** — QIR backend for direct quantum hardware compilation
- [ ] **v1.0** — Production release with complete standard library

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Good first issues are labeled `good first issue` — these are scoped, approachable tasks for newcomers. Areas where help is especially welcome:

- **Standard library functions** — implementing quantum algorithms (Grover, Shor, VQE variants)
- **Hardware profiles** — adding new QPU specifications
- **Type checker improvements** — better error messages, more inference
- **Documentation** — tutorials, examples, translations
- **Testing** — edge cases, property-based tests

## Community

- Website: [synaphe.io](https://synaphe.io)
- Discussions: GitHub Discussions
- Issues: GitHub Issues

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

## Citation

If you use Synaphe in research, please cite:

```bibtex
@software{synaphe2026,
  title={Synaphe: A Programming Language for Hybrid AI and Quantum Computing},
  year={2026},
  url={https://github.com/martus-spinther/synaphe}
}
```
