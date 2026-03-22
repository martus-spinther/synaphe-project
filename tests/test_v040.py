#!/usr/bin/env python3
"""
Synaphe v0.4.0 Tests — QAD Gradient System & Source Maps
Addresses the three criticisms from Google AI Mode.
"""

import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stdlib.qad import (
    quantum_grad, select_gradient_method, GradMethod,
    standard_parameter_shift, generalized_parameter_shift,
    stochastic_parameter_shift, hadamard_test_gradient,
    finite_difference_gradient
)
from src.sourcemap import (
    SourceMap, emit_with_source_map, format_synaphe_error,
    install_source_map_hook, uninstall_source_map_hook
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


# ═══════════════════════════════════════════════════════════════════
# CRITICISM 1 FIX: Source Maps (Black Box Problem)
# ═══════════════════════════════════════════════════════════════════

def test_source_map_creation():
    """Source map should map Python lines to Synaphe lines."""
    sm = SourceMap(synaphe_file="test.synaphe", synaphe_source="let x = 42")
    sm.add(python_line=5, synaphe_line=1, synaphe_source="let x = 42")
    mapping = sm.lookup(5)
    assert mapping is not None
    assert mapping.synaphe_line == 1
    assert mapping.synaphe_source == "let x = 42"

def test_source_map_nearest_lookup():
    """Lookup should find nearest preceding mapping."""
    sm = SourceMap(synaphe_file="test.synaphe", synaphe_source="")
    sm.add(python_line=1, synaphe_line=1, synaphe_source="let x = 1")
    sm.add(python_line=10, synaphe_line=5, synaphe_source="let y = 2")
    # Line 7 should map to the L1 mapping (nearest preceding)
    mapping = sm.lookup(7)
    assert mapping is not None
    assert mapping.synaphe_line == 1

def test_emit_with_source_map():
    """Transpiled code should include source map comments."""
    python_code = "x = 42\ny = x + 1"
    synaphe_source = "let x = 42\nlet y = x + 1"
    annotated, sm = emit_with_source_map(python_code, synaphe_source)
    assert "synaphe:L" in annotated

def test_format_error_with_context():
    """Error formatting should show Synaphe source context."""
    sm = SourceMap(synaphe_file="test.synaphe",
                   synaphe_source="let x = 42\nlet y = x @ w\nprint(y)")
    sm.add(5, 2, 0, "let y = x @ w")
    msg = format_synaphe_error(ValueError("shape mismatch"), sm, 5)
    assert "test.synaphe" in msg
    assert "x @ w" in msg

def test_exception_hook_installs():
    """The custom exception hook should install and uninstall cleanly."""
    original = sys.excepthook
    install_source_map_hook()
    assert sys.excepthook != original
    uninstall_source_map_hook()
    assert sys.excepthook == sys.__excepthook__


# ═══════════════════════════════════════════════════════════════════
# CRITICISM 2 FIX: Multiple Gradient Methods (Limited Flexibility)
# ═══════════════════════════════════════════════════════════════════

# Test function: f(x) = cos(x), so f'(x) = -sin(x)
def cos_fn(x):
    if isinstance(x, (list, tuple)):
        return math.cos(x[0])
    return math.cos(x)

def test_standard_psr_accuracy():
    """Standard PSR should compute correct gradient for cos(x)."""
    grads = standard_parameter_shift(cos_fn, [0.5])
    expected = -math.sin(0.5)
    assert abs(grads[0] - expected) < 0.01, f"Got {grads[0]}, expected {expected}"

def test_generalized_psr_accuracy():
    """Generalized PSR should match standard PSR for single-frequency gates."""
    grads = generalized_parameter_shift(cos_fn, [0.5], frequencies=[1.0])
    expected = -math.sin(0.5)
    assert abs(grads[0] - expected) < 0.05, f"Got {grads[0]}, expected {expected}"

def test_stochastic_psr_accuracy():
    """Stochastic PSR should approximate the gradient (higher variance)."""
    grads = stochastic_parameter_shift(cos_fn, [0.5], n_samples=50)
    expected = -math.sin(0.5)
    assert abs(grads[0] - expected) < 0.15, f"Got {grads[0]}, expected {expected}"

def test_hadamard_test_accuracy():
    """Hadamard test should approximate the gradient."""
    grads = hadamard_test_gradient(cos_fn, [0.5])
    expected = -math.sin(0.5)
    assert abs(grads[0] - expected) < 0.01, f"Got {grads[0]}, expected {expected}"

def test_finite_diff_accuracy():
    """Finite differences should approximate the gradient."""
    grads = finite_difference_gradient(cos_fn, [0.5])
    expected = -math.sin(0.5)
    assert abs(grads[0] - expected) < 0.001, f"Got {grads[0]}, expected {expected}"

def test_auto_method_selection_standard():
    """Auto should select standard PSR for RX/RY/RZ gates."""
    method = select_gradient_method("RX")
    assert method == GradMethod.STANDARD_SHIFT

def test_auto_method_selection_generalized():
    """Auto should select generalized PSR for multi-eigenvalue gates."""
    method = select_gradient_method("OrbitalRotation")
    assert method == GradMethod.GENERALIZED_SHIFT

def test_auto_method_selection_stochastic():
    """Auto should select stochastic PSR for unknown spectrum gates."""
    method = select_gradient_method("HamiltonianEvolution")
    assert method == GradMethod.STOCHASTIC_SHIFT

def test_auto_method_selection_unknown():
    """Unknown gates should default to stochastic PSR."""
    method = select_gradient_method("MyCustomGate")
    assert method == GradMethod.STOCHASTIC_SHIFT

def test_unified_grad_api():
    """quantum_grad() should work as a unified API."""
    grad_fn = quantum_grad(cos_fn, method="standard_shift")
    grads = grad_fn(0.5)
    expected = -math.sin(0.5)
    assert abs(grads[0] - expected) < 0.01

def test_unified_grad_auto():
    """quantum_grad(method='auto') should select and compute correctly."""
    grad_fn = quantum_grad(cos_fn, method="auto")
    grads = grad_fn(0.5)
    expected = -math.sin(0.5)
    assert abs(grads[0] - expected) < 0.01

def test_unified_grad_invalid_method():
    """Invalid method name should raise ValueError."""
    grad_fn = quantum_grad(cos_fn, method="nonexistent")
    try:
        grad_fn(0.5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown gradient method" in str(e)

def test_multiparameter_gradient():
    """Gradient should work for functions of multiple parameters."""
    def multi_fn(params):
        return math.cos(params[0]) + math.sin(params[1])

    grads = standard_parameter_shift(multi_fn, [0.5, 0.3])
    assert abs(grads[0] - (-math.sin(0.5))) < 0.01
    assert abs(grads[1] - math.cos(0.3)) < 0.01


# ═══════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\n\033[1m🔬 Synaphe v0.4.0 — Addressing the Three Criticisms\033[0m\n")

    print("\033[1mCriticism 1 Fix — Source Maps (Black Box):\033[0m")
    test("source map creation", test_source_map_creation)
    test("nearest-line lookup", test_source_map_nearest_lookup)
    test("emit with source map comments", test_emit_with_source_map)
    test("error formatting with context", test_format_error_with_context)
    test("exception hook installs cleanly", test_exception_hook_installs)

    print(f"\n\033[1mCriticism 2 Fix — Multiple Gradient Methods:\033[0m")
    test("standard PSR: cos'(0.5) = -sin(0.5)", test_standard_psr_accuracy)
    test("generalized PSR matches standard", test_generalized_psr_accuracy)
    test("stochastic PSR approximates gradient", test_stochastic_psr_accuracy)
    test("Hadamard test approximates gradient", test_hadamard_test_accuracy)
    test("finite differences approximate gradient", test_finite_diff_accuracy)
    test("auto selects standard for RX", test_auto_method_selection_standard)
    test("auto selects generalized for OrbitalRotation", test_auto_method_selection_generalized)
    test("auto selects stochastic for HamiltonianEvolution", test_auto_method_selection_stochastic)
    test("auto selects stochastic for unknown gates", test_auto_method_selection_unknown)
    test("unified grad API with explicit method", test_unified_grad_api)
    test("unified grad API with auto method", test_unified_grad_auto)
    test("invalid method raises ValueError", test_unified_grad_invalid_method)
    test("multi-parameter gradient", test_multiparameter_gradient)

    print(f"\n{'='*55}")
    print(f"\033[1mResults: {tests_passed}/{tests_run} passed\033[0m", end="")
    if tests_passed == tests_run:
        print(f" \033[32m— ALL PASS ✓\033[0m")
    else:
        print(f" \033[31m— {tests_run - tests_passed} FAILED\033[0m")
    print()
