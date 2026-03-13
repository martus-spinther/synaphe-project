#!/usr/bin/env python3
"""
Synaphe Type Checker Tests

Tests the three pillars:
1. Tensor shape safety (catches the #1 PyTorch runtime error)
2. Linear quantum safety (enforces no-cloning theorem)
3. Differentiable tracking (grad() across classical-quantum boundary)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse
from src.typechecker import (
    TypeChecker, typecheck,
    ShapeMismatchError, LinearityError, GradientError
)
from src.types import (
    TensorType, DimLit, DimVar, QStateType, QRegisterType,
    MeasurementType, GradType, FloatType, IntType,
    matmul_shape, broadcast_shape, ShapeError
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
tests_run = 0
tests_passed = 0

def test(name, fn):
    global tests_run, tests_passed
    tests_run += 1
    try:
        fn()
        tests_passed += 1
        print(f"  {PASS} {name}")
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")


# ════════════════════════════════════════════════════════════════════
# PILLAR 1: TENSOR SHAPE SAFETY
# ════════════════════════════════════════════════════════════════════

def test_matmul_valid_shapes():
    """[32, 784] @ [784, 128] -> [32, 128] ✓"""
    a = TensorType(dtype="Float32", shape=(DimLit(32), DimLit(784)))
    b = TensorType(dtype="Float32", shape=(DimLit(784), DimLit(128)))
    result = matmul_shape(a, b)
    assert result.shape == (DimLit(32), DimLit(128)), f"Got {result.shape}"

def test_matmul_invalid_shapes():
    """[32, 784] @ [10, 128] -> ERROR (784 != 10) ✓"""
    a = TensorType(dtype="Float32", shape=(DimLit(32), DimLit(784)))
    b = TensorType(dtype="Float32", shape=(DimLit(10), DimLit(128)))
    try:
        matmul_shape(a, b)
        assert False, "Should have raised ShapeError"
    except ShapeError:
        pass  # Expected!

def test_matmul_symbolic_dims():
    """[batch, 784] @ [784, 128] -> [batch, 128] (symbolic dim passes)"""
    a = TensorType(dtype="Float32", shape=(DimVar("batch"), DimLit(784)))
    b = TensorType(dtype="Float32", shape=(DimLit(784), DimLit(128)))
    result = matmul_shape(a, b)
    assert len(result.shape) == 2
    assert isinstance(result.shape[0], DimVar)
    assert result.shape[0].name == "batch"

def test_broadcast_same_shape():
    """[32, 10] + [32, 10] -> [32, 10]"""
    a = TensorType(dtype="Float32", shape=(DimLit(32), DimLit(10)))
    b = TensorType(dtype="Float32", shape=(DimLit(32), DimLit(10)))
    result = broadcast_shape(a, b)
    assert result.shape == (DimLit(32), DimLit(10))

def test_broadcast_with_1():
    """[32, 10] + [1, 10] -> [32, 10] (broadcast)"""
    a = TensorType(dtype="Float32", shape=(DimLit(32), DimLit(10)))
    b = TensorType(dtype="Float32", shape=(DimLit(1), DimLit(10)))
    result = broadcast_shape(a, b)
    assert result.shape == (DimLit(32), DimLit(10))

def test_broadcast_incompatible():
    """[32, 10] + [64, 10] -> ERROR (32 != 64, neither is 1)"""
    a = TensorType(dtype="Float32", shape=(DimLit(32), DimLit(10)))
    b = TensorType(dtype="Float32", shape=(DimLit(64), DimLit(10)))
    try:
        broadcast_shape(a, b)
        assert False, "Should have raised ShapeError"
    except ShapeError:
        pass

def test_typechecker_catches_matmul_mismatch():
    """Type checker catches shape mismatch in actual Nova code."""
    source = """
let x: Tensor<Float32, [32, 784]> = randn(32, 784)
let w: Tensor<Float32, [10, 128]> = randn(10, 128)
let y = x @ w
"""
    errors, warnings = typecheck(parse(source))
    shape_errors = [e for e in errors if isinstance(e, ShapeMismatchError)]
    assert len(shape_errors) > 0, f"Expected shape error, got: {errors}"

def test_typechecker_passes_valid_matmul():
    """Type checker accepts valid matmul shapes."""
    source = """
let x: Tensor<Float32, [32, 784]> = randn(32, 784)
let w: Tensor<Float32, [784, 128]> = randn(784, 128)
let y = x @ w
"""
    errors, warnings = typecheck(parse(source))
    shape_errors = [e for e in errors if isinstance(e, ShapeMismatchError)]
    assert len(shape_errors) == 0, f"Unexpected errors: {shape_errors}"


# ════════════════════════════════════════════════════════════════════
# PILLAR 2: LINEAR QUANTUM SAFETY
# ════════════════════════════════════════════════════════════════════

def test_qubit_single_use_ok():
    """Using a qubit once is fine."""
    source = """
let q = qregister(4)
let result = measure(q)
"""
    errors, warnings = typecheck(parse(source))
    linearity_errors = [e for e in errors if isinstance(e, LinearityError)]
    assert len(linearity_errors) == 0, f"Unexpected: {linearity_errors}"

def test_qubit_reuse_after_measure():
    """Using a qubit after measurement should ERROR (No-Cloning Theorem)."""
    source = """
let q = qregister(4)
let classical = measure(q)
let oops = hadamard(q)
"""
    errors, warnings = typecheck(parse(source))
    linearity_errors = [e for e in errors if isinstance(e, LinearityError)]
    assert len(linearity_errors) > 0, (
        f"Expected linearity error for reuse after measure, got: {errors}"
    )

def test_qubit_double_consume():
    """Passing a quantum state to two functions should ERROR."""
    source = """
let q = qregister(4)
let a = hadamard(q)
let b = rx(q)
"""
    errors, warnings = typecheck(parse(source))
    linearity_errors = [e for e in errors if isinstance(e, LinearityError)]
    assert len(linearity_errors) > 0, (
        f"Expected linearity error for double consumption, got: {errors}"
    )

def test_measure_produces_classical():
    """After measurement, result is classical and CAN be copied."""
    source = """
let q = qregister(4)
let bits = measure(q)
let a = bits
let b = bits
"""
    errors, warnings = typecheck(parse(source))
    # bits is classical (MeasurementType), so copying is fine
    linearity_errors = [e for e in errors if isinstance(e, LinearityError)]
    assert len(linearity_errors) == 0, f"Unexpected: {linearity_errors}"

def test_unused_qubit_warning():
    """Allocating a qubit but never using it should WARN."""
    source = """
let q = qregister(4)
let x = 42
"""
    errors, warnings = typecheck(parse(source))
    qubit_warnings = [w for w in warnings if "never used" in w.lower() or "qubit" in w.lower() or "quantum" in w.lower()]
    assert len(qubit_warnings) > 0, f"Expected unused qubit warning, got warnings: {warnings}"


# ════════════════════════════════════════════════════════════════════
# PILLAR 3: DIFFERENTIABLE TRACKING
# ════════════════════════════════════════════════════════════════════

def test_grad_returns_grad_type():
    """grad(fn) should return Grad<T> type."""
    source = """
fn loss(x: Float) -> Float {
    return x * x
}
let g = grad(loss)
"""
    errors, warnings = typecheck(parse(source))
    # Should compile without errors
    assert not any(isinstance(e, GradientError) for e in errors), f"Unexpected: {errors}"

def test_grad_needs_function():
    """grad() without argument should error."""
    source = """
let g = grad()
"""
    errors, warnings = typecheck(parse(source))
    grad_errors = [e for e in errors if isinstance(e, GradientError)]
    assert len(grad_errors) > 0, f"Expected gradient error, got: {errors}"


# ════════════════════════════════════════════════════════════════════
# INTEGRATION: THE BRIDGE SIGNATURE
# ════════════════════════════════════════════════════════════════════

def test_bridge_function_signature():
    """
    The north star: can the type checker understand a hybrid function?
    fn train_step(data: Tensor<f32, [4]>, state: QState<4>) -> Float
    """
    source = """
fn train_step(data: Tensor<Float32, [4]>, state: QState) -> Float {
    let prediction = data |> softmax
    let q_result = measure(state)
    return 0.0
}
"""
    errors, warnings = typecheck(parse(source))
    # The function should type-check: data is classical, state is quantum,
    # state gets measured (consumed), return is classical
    fatal = [e for e in errors if isinstance(e, (ShapeMismatchError, LinearityError))]
    assert len(fatal) == 0, f"Bridge function failed: {fatal}"

def test_hybrid_pipeline():
    """Pipeline that crosses the classical-quantum boundary."""
    source = """
let q = qregister(4)
let result = q |> hadamard |> measure
"""
    errors, warnings = typecheck(parse(source))
    linearity_errors = [e for e in errors if isinstance(e, LinearityError)]
    assert len(linearity_errors) == 0, f"Unexpected: {linearity_errors}"

def test_golden_snippet_vqe_structure():
    """The VQE golden snippet structure should type-check."""
    source = """
let H = hamiltonian("H2", basis="sto-3g")

fn ansatz(theta: Float) -> Float {
    let q = qregister(4)
    let evolved = q |> hadamard |> rx
    let measured = measure(evolved)
    return 0.0
}

let result = minimize(ansatz)
"""
    errors, warnings = typecheck(parse(source))
    fatal = [e for e in errors if isinstance(e, (ShapeMismatchError, LinearityError))]
    assert len(fatal) == 0, f"VQE structure failed: {fatal}"


# ════════════════════════════════════════════════════════════════════
# TYPE REPRESENTATION TESTS
# ════════════════════════════════════════════════════════════════════

def test_type_repr():
    """Types should have readable string representations."""
    assert repr(IntType()) == "Int"
    assert repr(FloatType(bits=32)) == "Float32"
    assert repr(TensorType(dtype="Float32", shape=(DimLit(32), DimLit(784)))) == "Tensor<Float32, [32, 784]>"
    assert repr(QRegisterType(n_qubits=4)) == "QRegister<4>"
    assert repr(QStateType(n_qubits=4)) == "QState<4>"
    assert repr(MeasurementType(n_bits=4)) == "Measurement<4>"
    assert repr(GradType(inner_type=FloatType(bits=32))) == "Grad<Float32>"


# ════════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n\033[1m🔬 Synaphe Type Checker Tests\033[0m\n")

    print("\033[1mPillar 1 — Tensor Shape Safety:\033[0m")
    test("valid matmul [32,784] @ [784,128]", test_matmul_valid_shapes)
    test("invalid matmul [32,784] @ [10,128] → ERROR", test_matmul_invalid_shapes)
    test("symbolic dims [batch,784] @ [784,128]", test_matmul_symbolic_dims)
    test("broadcast [32,10] + [32,10]", test_broadcast_same_shape)
    test("broadcast [32,10] + [1,10]", test_broadcast_with_1)
    test("broadcast [32,10] + [64,10] → ERROR", test_broadcast_incompatible)
    test("checker catches matmul mismatch in code", test_typechecker_catches_matmul_mismatch)
    test("checker passes valid matmul in code", test_typechecker_passes_valid_matmul)

    print(f"\n\033[1mPillar 2 — Linear Quantum Safety:\033[0m")
    test("single qubit use OK", test_qubit_single_use_ok)
    test("reuse after measure → ERROR (No-Cloning)", test_qubit_reuse_after_measure)
    test("double consume → ERROR (linearity)", test_qubit_double_consume)
    test("measurement result is classical (copyable)", test_measure_produces_classical)
    test("unused qubit → WARNING", test_unused_qubit_warning)

    print(f"\n\033[1mPillar 3 — Differentiable Tracking:\033[0m")
    test("grad(fn) returns Grad<T>", test_grad_returns_grad_type)
    test("grad() without arg → ERROR", test_grad_needs_function)

    print(f"\n\033[1mIntegration — The Bridge:\033[0m")
    test("hybrid fn(Tensor, QState) → Float", test_bridge_function_signature)
    test("pipeline crossing quantum-classical", test_hybrid_pipeline)
    test("VQE golden snippet structure", test_golden_snippet_vqe_structure)

    print(f"\n\033[1mType Representations:\033[0m")
    test("readable type repr strings", test_type_repr)

    # Summary
    print(f"\n{'='*55}")
    print(f"\033[1mResults: {tests_passed}/{tests_run} passed\033[0m", end="")
    if tests_passed == tests_run:
        print(f" \033[32m— ALL PASS ✓\033[0m")
    else:
        print(f" \033[31m— {tests_run - tests_passed} FAILED\033[0m")
    print()
