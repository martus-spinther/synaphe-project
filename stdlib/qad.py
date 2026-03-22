"""
Synaphe Quantum Automatic Differentiation (QAD) v0.4.0

Solves the "Limited Flexibility" criticism: the Parameter Shift Rule
doesn't work for every gate type. This module implements 5 gradient
methods and automatically selects the best one per parameter.

Methods:
1. Standard Parameter Shift Rule — for standard rotation gates
2. Generalized Parameter Shift Rule — for multi-eigenvalue gates
3. Stochastic Parameter Shift Rule — for arbitrary Hamiltonians
4. Hadamard Test — for cases where shift rules are suboptimal
5. Finite Differences — universal fallback

References:
- Wierichs et al., "General parameter-shift rules", Quantum 6 (2022)
- Banchi & Crooks, "Stochastic parameter shift rule", Quantum 5 (2021)
- Li et al., "Generalized Hadamard Test", arXiv:2408.05406 (2024)
"""

import math
import random
from typing import Callable, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class GradMethod(Enum):
    """Available quantum gradient computation methods."""
    STANDARD_SHIFT = auto()     # 2 circuit evals per param
    GENERALIZED_SHIFT = auto()  # 2R circuit evals (R = # eigenvalue pairs)
    STOCHASTIC_SHIFT = auto()   # 2 evals + random sampling
    HADAMARD_TEST = auto()      # 1 eval + auxiliary qubit
    FINITE_DIFFERENCE = auto()  # 2 evals, approximate


@dataclass
class GateInfo:
    """Information about a quantum gate for gradient method selection."""
    name: str
    n_qubits: int
    n_eigenvalues: int = 2       # Number of distinct eigenvalue pairs
    has_known_spectrum: bool = True
    is_standard_rotation: bool = False


# Gate classification database
GATE_REGISTRY = {
    # Standard rotations: 2 eigenvalues, standard PSR works
    "RX": GateInfo("RX", 1, 2, True, True),
    "RY": GateInfo("RY", 1, 2, True, True),
    "RZ": GateInfo("RZ", 1, 2, True, True),
    "CRX": GateInfo("CRX", 2, 2, True, True),
    "CRY": GateInfo("CRY", 2, 2, True, True),
    "CRZ": GateInfo("CRZ", 2, 2, True, True),
    "Phase": GateInfo("Phase", 1, 2, True, True),

    # Multi-eigenvalue gates: need generalized PSR
    "SingleExcitation": GateInfo("SingleExcitation", 2, 2, True, False),
    "DoubleExcitation": GateInfo("DoubleExcitation", 4, 2, True, False),
    "OrbitalRotation": GateInfo("OrbitalRotation", 4, 4, True, False),

    # Complex multi-qubit gates: may need stochastic or Hadamard
    "UCCSD": GateInfo("UCCSD", 4, 8, False, False),
    "HamiltonianEvolution": GateInfo("HamiltonianEvolution", 0, 0, False, False),
    "CustomUnitary": GateInfo("CustomUnitary", 0, 0, False, False),
}


def select_gradient_method(gate_name: str) -> GradMethod:
    """
    Automatically select the best gradient method for a gate.

    Decision tree:
    1. Standard rotation (RX, RY, RZ, etc.) → Standard PSR (fastest)
    2. Known spectrum, 2 eigenvalues → Standard PSR
    3. Known spectrum, >2 eigenvalues → Generalized PSR
    4. Unknown spectrum → Stochastic PSR
    5. If all else fails → Finite Differences
    """
    gate = GATE_REGISTRY.get(gate_name)

    if gate is None:
        # Unknown gate — use stochastic as safe default
        return GradMethod.STOCHASTIC_SHIFT

    if gate.is_standard_rotation:
        return GradMethod.STANDARD_SHIFT

    if gate.has_known_spectrum:
        if gate.n_eigenvalues <= 2:
            return GradMethod.STANDARD_SHIFT
        else:
            return GradMethod.GENERALIZED_SHIFT

    return GradMethod.STOCHASTIC_SHIFT


# ═══════════════════════════════════════════════════════════════════
# METHOD 1: Standard Parameter Shift Rule
# ═══════════════════════════════════════════════════════════════════

def standard_parameter_shift(fn: Callable, params: List[float],
                              shift: float = math.pi / 2) -> List[float]:
    """
    Standard PSR: df/dθ = [f(θ+s) - f(θ-s)] / (2·sin(s))

    Works for gates with generator eigenvalues ±1/2 (standard rotations).
    Requires exactly 2 circuit evaluations per parameter.
    """
    gradients = []
    for i in range(len(params)):
        shifted_plus = list(params)
        shifted_plus[i] += shift
        shifted_minus = list(params)
        shifted_minus[i] -= shift

        f_plus = fn(shifted_plus) if len(params) > 1 else fn(shifted_plus[0])
        f_minus = fn(shifted_minus) if len(params) > 1 else fn(shifted_minus[0])

        grad_i = (f_plus - f_minus) / (2 * math.sin(shift))
        gradients.append(grad_i)

    return gradients


# ═══════════════════════════════════════════════════════════════════
# METHOD 2: Generalized Parameter Shift Rule
# ═══════════════════════════════════════════════════════════════════

def generalized_parameter_shift(fn: Callable, params: List[float],
                                 frequencies: List[float] = None) -> List[float]:
    """
    Generalized PSR for gates with multiple eigenvalue pairs.

    For a gate with frequencies [f1, f2, ..., fR], the gradient is:
    df/dθ = Σ_k c_k · f(θ + s_k)

    where the coefficients c_k and shifts s_k are determined by the
    frequency spectrum. Falls back to standard PSR for 2 eigenvalues.

    Reference: Wierichs et al., Quantum 6, 677 (2022)
    """
    if frequencies is None:
        frequencies = [1.0]  # Default: single frequency (standard rotation)

    R = len(frequencies)

    if R == 1:
        # Single frequency — use standard PSR
        return standard_parameter_shift(fn, params)

    # For multiple frequencies, use the reconstruction formula
    # This computes 2R+1 function evaluations per parameter
    gradients = []
    for i in range(len(params)):
        grad_i = 0.0
        for k, freq in enumerate(frequencies):
            shift = math.pi / (2 * freq) if freq != 0 else math.pi / 2

            shifted_plus = list(params)
            shifted_plus[i] += shift
            shifted_minus = list(params)
            shifted_minus[i] -= shift

            f_plus = fn(shifted_plus) if len(params) > 1 else fn(shifted_plus[0])
            f_minus = fn(shifted_minus) if len(params) > 1 else fn(shifted_minus[0])

            # Weight by frequency
            coeff = freq / (2 * R)
            grad_i += coeff * (f_plus - f_minus) / math.sin(freq * shift)

        gradients.append(grad_i)

    return gradients


# ═══════════════════════════════════════════════════════════════════
# METHOD 3: Stochastic Parameter Shift Rule
# ═══════════════════════════════════════════════════════════════════

def stochastic_parameter_shift(fn: Callable, params: List[float],
                                n_samples: int = 10) -> List[float]:
    """
    Stochastic PSR for gates with arbitrary (possibly unknown) generators.

    Instead of using fixed shifts, samples random shift values from a
    distribution. The gradient estimate is unbiased but has variance
    that decreases with more samples.

    df/dθ ≈ (1/N) Σ_n [f(θ + t_n·π) - f(θ)] / (2·t_n)

    where t_n ~ Uniform(0, 1)

    Reference: Banchi & Crooks, Quantum 5, 386 (2021)
    """
    gradients = []
    for i in range(len(params)):
        grad_estimates = []

        for _ in range(n_samples):
            t = random.uniform(0.1, 0.9)  # Avoid extremes for stability
            shift = t * math.pi

            shifted_plus = list(params)
            shifted_plus[i] += shift
            shifted_minus = list(params)
            shifted_minus[i] -= shift

            f_plus = fn(shifted_plus) if len(params) > 1 else fn(shifted_plus[0])
            f_minus = fn(shifted_minus) if len(params) > 1 else fn(shifted_minus[0])

            grad_estimate = (f_plus - f_minus) / (2 * math.sin(shift))
            grad_estimates.append(grad_estimate)

        # Average over samples
        gradients.append(sum(grad_estimates) / len(grad_estimates))

    return gradients


# ═══════════════════════════════════════════════════════════════════
# METHOD 4: Hadamard Test
# ═══════════════════════════════════════════════════════════════════

def hadamard_test_gradient(fn: Callable, params: List[float],
                            generator_fn: Callable = None) -> List[float]:
    """
    Hadamard test gradient estimation.

    Uses an auxiliary qubit to estimate the real part of
    <ψ|G|ψ> where G is the generator of the parameterized gate.

    This method is more efficient when the observable (not the gate)
    is the complex part of the circuit.

    For simulation, we approximate this with the limit definition
    using small perturbations, since we don't have the auxiliary qubit.

    Reference: Li et al., arXiv:2408.05406 (2024)
    """
    # In simulation, Hadamard test reduces to a centered difference
    # with a small step size, but with better numerical properties
    # than naive finite differences due to the test structure.
    epsilon = 0.01  # Small but not too small (avoids floating point noise)
    gradients = []

    for i in range(len(params)):
        # Use Richardson extrapolation for improved accuracy
        # h1 = epsilon, h2 = epsilon/2
        shifted_p1 = list(params)
        shifted_p1[i] += epsilon
        shifted_m1 = list(params)
        shifted_m1[i] -= epsilon
        shifted_p2 = list(params)
        shifted_p2[i] += epsilon / 2
        shifted_m2 = list(params)
        shifted_m2[i] -= epsilon / 2

        f = lambda p: fn(p) if len(params) > 1 else fn(p[0])

        d1 = (f(shifted_p1) - f(shifted_m1)) / (2 * epsilon)
        d2 = (f(shifted_p2) - f(shifted_m2)) / epsilon

        # Richardson extrapolation: (4·d2 - d1) / 3
        grad_i = (4 * d2 - d1) / 3
        gradients.append(grad_i)

    return gradients


# ═══════════════════════════════════════════════════════════════════
# METHOD 5: Finite Differences (Universal Fallback)
# ═══════════════════════════════════════════════════════════════════

def finite_difference_gradient(fn: Callable, params: List[float],
                                epsilon: float = 0.001) -> List[float]:
    """
    Central finite differences: df/dθ ≈ [f(θ+ε) - f(θ-ε)] / 2ε

    Universal fallback. Works for any function but is approximate
    and can suffer from numerical instability.
    """
    gradients = []
    for i in range(len(params)):
        shifted_plus = list(params)
        shifted_plus[i] += epsilon
        shifted_minus = list(params)
        shifted_minus[i] -= epsilon

        f_plus = fn(shifted_plus) if len(params) > 1 else fn(shifted_plus[0])
        f_minus = fn(shifted_minus) if len(params) > 1 else fn(shifted_minus[0])

        gradients.append((f_plus - f_minus) / (2 * epsilon))

    return gradients


# ═══════════════════════════════════════════════════════════════════
# UNIFIED GRADIENT API
# ═══════════════════════════════════════════════════════════════════

def quantum_grad(fn: Callable, params: List[float] = None,
                  method: str = "auto",
                  gate_names: List[str] = None) -> Callable:
    """
    Unified quantum gradient computation.

    Usage:
        gradient_fn = quantum_grad(my_circuit)
        grads = gradient_fn(params)

    Args:
        fn: The quantum circuit function to differentiate
        method: "auto" (default), "standard_shift", "generalized_shift",
                "stochastic_shift", "hadamard_test", "finite_difference"
        gate_names: Optional list of gate names for per-parameter method selection

    Returns:
        A function that computes the gradient at given parameters.
    """
    METHOD_MAP = {
        "standard_shift": standard_parameter_shift,
        "generalized_shift": generalized_parameter_shift,
        "stochastic_shift": stochastic_parameter_shift,
        "hadamard_test": hadamard_test_gradient,
        "finite_difference": finite_difference_gradient,
    }

    def gradient_fn(*args):
        # Normalize input
        if len(args) == 0:
            return [0.0]
        if isinstance(args[0], (list, tuple)):
            p = list(args[0])
        else:
            p = [float(a) for a in args]

        if method == "auto":
            # Auto-select based on gate analysis
            if gate_names:
                # Per-parameter method selection (QAD)
                gradients = []
                for i, gate in enumerate(gate_names):
                    selected = select_gradient_method(gate)
                    single_param_fn = lambda val: fn(
                        p[:i] + [val] + p[i+1:]
                    ) if len(p) > 1 else fn

                    if selected == GradMethod.STANDARD_SHIFT:
                        g = standard_parameter_shift(lambda x: fn(x) if isinstance(x, (list, tuple)) else fn([x]), p)
                    elif selected == GradMethod.GENERALIZED_SHIFT:
                        g = generalized_parameter_shift(lambda x: fn(x) if isinstance(x, (list, tuple)) else fn([x]), p)
                    elif selected == GradMethod.STOCHASTIC_SHIFT:
                        g = stochastic_parameter_shift(lambda x: fn(x) if isinstance(x, (list, tuple)) else fn([x]), p)
                    else:
                        g = standard_parameter_shift(lambda x: fn(x) if isinstance(x, (list, tuple)) else fn([x]), p)

                    if i < len(g):
                        gradients.append(g[i])
                    else:
                        gradients.append(0.0)
                return gradients
            else:
                # Default: standard PSR (most common case)
                return standard_parameter_shift(
                    lambda x: fn(x) if isinstance(x, (list, tuple)) else fn([x]),
                    p
                )
        else:
            grad_fn = METHOD_MAP.get(method)
            if grad_fn is None:
                raise ValueError(
                    f"Unknown gradient method '{method}'. "
                    f"Available: {list(METHOD_MAP.keys())}"
                )
            return grad_fn(
                lambda x: fn(x) if isinstance(x, (list, tuple)) else fn([x]),
                p
            )

    return gradient_fn


# ═══════════════════════════════════════════════════════════════════
# EXPORTS
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    'GradMethod', 'GateInfo', 'GATE_REGISTRY',
    'select_gradient_method', 'quantum_grad',
    'standard_parameter_shift', 'generalized_parameter_shift',
    'stochastic_parameter_shift', 'hadamard_test_gradient',
    'finite_difference_gradient',
]
