# Synaphe Design Principles — Revised After Golden Snippet Test

## What the Golden Snippets Proved

After writing 5 real-world problems (VQE, hybrid ML, QAOA, differentiable 
chemistry, anomaly detection) in both Python and Synaphe, here's what we learned:

### 1. The Syntax Works

The pipeline operator `|>` is the star of the language. Every single golden 
snippet uses it as the primary way to express computation flow. It reads 
naturally left-to-right and eliminates the nested function call mess that 
plagues Python quantum code.

**Before (Python):** `qml.expval(qml.PauliZ(circuit(params)))`
**After (Synaphe):** `circuit(params) |> expectation(PauliZ)`

### 2. The "Lock Doors" Principle (from Elo)

Google AI Mode's insight about Elo's non-Turing-completeness translated into 
a design principle we're calling **"lock doors"**:

- Tensor shapes are checked at compile time → no `RuntimeError: size mismatch`
- Quantum register sizes are part of the type → `QRegister<4>` can't be passed 
  where `QRegister<8>` is expected
- Schema constraints are validated at data boundaries → bad data never reaches 
  your model
- Hardware targets are declared, not coded → you can't accidentally run quantum 
  code on a CPU

**Key insight:** The language should make the *wrong thing impossible*, not just 
the right thing easy.

### 3. The Unified Container Insight

Google AI Mode's observation that tensors and quantum states are both 
"containers over probability spaces" led to this design decision:

```flux
// Both of these are the same fundamental type:
let classical: Tensor<Float32, [batch, 128]>  // Probability distribution over features
let quantum: QState<4>                         // Probability distribution over basis states

// And they convert naturally:
let features = quantum_state |> measure |> to_tensor
let encoded = tensor |> encode_amplitude |> as_qstate
```

This means a function like `expectation()` works on both classical and quantum
inputs. The compiler dispatches to the right implementation.

### 4. What Still Needs Work

From the golden snippet exercise, the language needs these additions:

#### a. Range syntax: `0..4`
Used in every quantum snippet for qubit iteration. Need to add to lexer/parser.

#### b. Anonymous functions in expressions: `fn(x) => expr`
Used for inline cost functions in VQE and QAOA. The parser supports Lambda 
in the AST but we need syntax sugar.

#### c. String interpolation: `"Energy: {value} Ha"`
Every snippet uses this. Need f-string style interpolation in the transpiler.

#### d. Streaming types: `Stream<T>`
The anomaly detection snippet needs lazy evaluation and streaming. This is a 
Phase 2 feature but the type system should support it from day one.

#### e. The `@target(auto)` system
The hardware dispatch decorator is referenced but not implemented. This is the 
core promise of the language — it needs to be real.

## Revised Roadmap

### Phase 1: Make the Golden Snippets Actually Compile (Now)
- [ ] Range operator `0..n`
- [ ] Lambda syntax `fn(x) => expr`
- [ ] String interpolation `"{expr}"`
- [ ] Type checker for tensor shapes
- [ ] Standard library: `hamiltonian()`, `minimize()`, `expectation()`
- [ ] All 5 golden snippets parse and transpile correctly

### Phase 2: Make It Run (1-3 months)
- [ ] Python runtime library (`flux_runtime`) that provides quantum/tensor ops
- [ ] @target decorator dispatches to PyTorch/PennyLane/Qiskit
- [ ] REPL with visualization of quantum states and tensor shapes
- [ ] Package manager (`flux add pennylane`)

### Phase 3: Make It Fast (3-6 months)
- [ ] Compile-time tensor shape verification (the real type checker)
- [ ] MLIR backend for classical code
- [ ] QIR backend for quantum code
- [ ] Whole-program optimization across classical/quantum boundaries

### Phase 4: Make It Standard (6-12 months)
- [ ] Language server protocol (LSP) for IDE support
- [ ] Jupyter kernel for notebook integration
- [ ] Published specification and formal grammar
- [ ] Community standard library contributions

## The Thesis (Refined)

> **Write quantum-classical hybrid code in 18 lines that would take 65 in Python.
> If the shapes don't match, it doesn't compile. If it compiles, it runs 
> anywhere — CPU, GPU, or QPU.**

This is not about replacing Python. It's about being the layer above Python that 
makes hybrid AI-quantum work safe, readable, and portable.
