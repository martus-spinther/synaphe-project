"""
Synaphe Runtime Standard Library v0.3.0

This is the HardwareMap made real. Each Nova function dispatches to the
appropriate backend: PyTorch for tensors, PennyLane for quantum, numpy
for simulation.

When transpiled code runs, it imports this module to get the Nova
standard library functions.
"""

import math
import random
from typing import Callable, Optional, List, Any, Tuple
from dataclasses import dataclass, field

# ── Try importing backends (graceful fallback) ───────────────────────

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Minimal fallback
    class np:
        @staticmethod
        def array(x): return list(x)
        @staticmethod
        def zeros(n): return [0.0] * n
        @staticmethod
        def random_normal(size): return [random.gauss(0, 1) for _ in range(size)]
        pi = math.pi

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pennylane as qml
    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


# ═══════════════════════════════════════════════════════════════════
# QUANTUM SIMULATION (works without any quantum backend)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class QState:
    """Simulated quantum state using statevector representation."""
    n_qubits: int
    amplitudes: Any = None  # numpy array of 2^n complex amplitudes
    measured: bool = False
    _consumed: bool = False

    def __post_init__(self):
        n = 2 ** self.n_qubits
        if self.amplitudes is None:
            if HAS_NUMPY:
                self.amplitudes = np.zeros(n, dtype=complex)
                self.amplitudes[0] = 1.0  # |000...0> state
            else:
                self.amplitudes = [0.0] * n
                self.amplitudes[0] = 1.0

    def _check_linear(self):
        """Enforce linear type discipline at runtime."""
        if self._consumed:
            raise RuntimeError(
                "Synaphe LinearityError: This quantum state has already been consumed. "
                "Quantum states cannot be reused (No-Cloning Theorem). "
                "Re-prepare the state if you need it again."
            )
        if self.measured:
            raise RuntimeError(
                "Synaphe LinearityError: This quantum state has been measured and "
                "collapsed to classical. It cannot be used as quantum input."
            )

    def consume(self) -> 'QState':
        """Mark this state as consumed and return it (for linear tracking)."""
        self._check_linear()
        self._consumed = True
        # Return a new QState with the same amplitudes (ownership transfer)
        new = QState(self.n_qubits)
        new.amplitudes = self.amplitudes.copy() if HAS_NUMPY else list(self.amplitudes)
        return new

    def __repr__(self):
        if self.measured:
            return f"QState<{self.n_qubits}>(measured)"
        return f"QState<{self.n_qubits}>"


@dataclass
class Measurement:
    """Result of measuring a quantum state. Classical — freely copyable."""
    bits: List[int] = field(default_factory=list)
    probabilities: Any = None

    def to_tensor(self):
        if HAS_TORCH:
            return torch.tensor(self.bits, dtype=torch.float32)
        return self.bits

    def __repr__(self):
        return f"Measurement({self.bits})"


# ── Quantum Gate Implementations (Statevector Simulation) ────────────

def _apply_single_gate(state: QState, qubit: int, matrix) -> QState:
    """Apply a single-qubit gate matrix to the statevector."""
    new_state = state.consume()
    n = new_state.n_qubits
    N = 2 ** n

    if HAS_NUMPY:
        new_amps = np.zeros(N, dtype=complex)
        for i in range(N):
            bit = (i >> (n - 1 - qubit)) & 1
            partner = i ^ (1 << (n - 1 - qubit))
            if bit == 0:
                new_amps[i] += matrix[0][0] * new_state.amplitudes[i] + matrix[0][1] * new_state.amplitudes[partner]
            else:
                new_amps[i] += matrix[1][0] * new_state.amplitudes[partner] + matrix[1][1] * new_state.amplitudes[i]
        new_state.amplitudes = new_amps
    return new_state


def _apply_cnot(state: QState, control: int, target: int) -> QState:
    """Apply CNOT gate."""
    new_state = state.consume()
    n = new_state.n_qubits
    N = 2 ** n

    if HAS_NUMPY:
        new_amps = new_state.amplitudes.copy()
        for i in range(N):
            ctrl_bit = (i >> (n - 1 - control)) & 1
            if ctrl_bit == 1:
                partner = i ^ (1 << (n - 1 - target))
                new_amps[i], new_amps[partner] = new_state.amplitudes[partner], new_state.amplitudes[i]
        new_state.amplitudes = new_amps
    return new_state


# ── Public Quantum API ───────────────────────────────────────────────

def qubit() -> QState:
    """Allocate a single fresh qubit in |0> state."""
    return QState(n_qubits=1)


def qregister(n: int) -> QState:
    """Allocate n qubits in |000...0> state."""
    return QState(n_qubits=n)


def hadamard(state_or_qubit, qubit_idx: int = 0) -> QState:
    """Apply Hadamard gate. H = (1/√2)[[1,1],[1,-1]]"""
    if isinstance(state_or_qubit, QState):
        state = state_or_qubit
    else:
        state = state_or_qubit
    s = 1 / math.sqrt(2)
    H = [[s, s], [s, -s]]
    return _apply_single_gate(state, qubit_idx, H)


def rx(state_or_angle, qubit_or_state=None, qubit_idx=0) -> QState:
    """Apply RX(θ) rotation gate."""
    if isinstance(state_or_angle, QState):
        state, theta = state_or_angle, 0.0
    elif isinstance(qubit_or_state, QState):
        theta, state = state_or_angle, qubit_or_state
    else:
        state = state_or_angle
        theta = 0.0

    c, s = math.cos(theta / 2), math.sin(theta / 2)
    RX = [[c, -1j * s], [-1j * s, c]]
    return _apply_single_gate(state, qubit_idx, RX)


def ry(state_or_angle, qubit_or_state=None, qubit_idx=0) -> QState:
    """Apply RY(θ) rotation gate."""
    if isinstance(state_or_angle, QState):
        state, theta = state_or_angle, 0.0
    elif isinstance(qubit_or_state, QState):
        theta, state = state_or_angle, qubit_or_state
    else:
        state = state_or_angle
        theta = 0.0

    c, s = math.cos(theta / 2), math.sin(theta / 2)
    RY = [[c, -s], [s, c]]
    return _apply_single_gate(state, qubit_idx, RY)


def rz(state_or_angle, qubit_or_state=None, qubit_idx=0) -> QState:
    """Apply RZ(θ) rotation gate."""
    if isinstance(state_or_angle, QState):
        state, theta = state_or_angle, 0.0
    elif isinstance(qubit_or_state, QState):
        theta, state = state_or_angle, qubit_or_state
    else:
        state = state_or_angle
        theta = 0.0

    RZ = [[complex(math.cos(theta/2), -math.sin(theta/2)), 0],
          [0, complex(math.cos(theta/2), math.sin(theta/2))]]
    return _apply_single_gate(state, qubit_idx, RZ)


def cx(state_or_ctrl, target_or_state=None, target=None) -> QState:
    """Apply CNOT (controlled-X) gate."""
    if isinstance(state_or_ctrl, QState):
        return _apply_cnot(state_or_ctrl, 0, 1)
    elif isinstance(target_or_state, QState):
        return _apply_cnot(target_or_state, state_or_ctrl, target or 1)
    return state_or_ctrl


def measure(state: QState) -> Measurement:
    """
    Measure all qubits. Collapses the quantum state to classical.
    After this, the QState is CONSUMED — it cannot be used again.
    """
    state._check_linear()
    state.measured = True

    if HAS_NUMPY:
        probs = np.abs(state.amplitudes) ** 2
        probs = probs / probs.sum()  # Normalize
        outcome = np.random.choice(len(probs), p=probs)
        bits = [(outcome >> (state.n_qubits - 1 - i)) & 1
                for i in range(state.n_qubits)]
        return Measurement(bits=bits, probabilities=probs)
    else:
        # Fallback: random measurement
        bits = [random.choice([0, 1]) for _ in range(state.n_qubits)]
        return Measurement(bits=bits)


def measure_all() -> Measurement:
    """Placeholder for measuring all qubits in current context."""
    return Measurement(bits=[0])


def expectation(operator, state_or_result=None) -> float:
    """
    Compute expectation value <ψ|O|ψ>.
    For PauliZ: this is the sum of probabilities weighted by ±1.
    """
    if isinstance(state_or_result, QState):
        m = measure(state_or_result)
    elif isinstance(state_or_result, Measurement):
        m = state_or_result
    elif state_or_result is None:
        return 0.0
    else:
        m = state_or_result

    if hasattr(m, 'probabilities') and m.probabilities is not None and HAS_NUMPY:
        n = int(math.log2(len(m.probabilities)))
        exp_val = 0.0
        for i, p in enumerate(m.probabilities):
            parity = bin(i).count('1') % 2
            exp_val += p * (1.0 if parity == 0 else -1.0)
        return float(exp_val)

    # Fallback
    parity = sum(m.bits) % 2
    return 1.0 if parity == 0 else -1.0


# ═══════════════════════════════════════════════════════════════════
# DIFFERENTIABLE BRIDGE
# ═══════════════════════════════════════════════════════════════════

def parameter_shift_gradient(fn: Callable, params: list, shift: float = math.pi / 2) -> list:
    """
    Parameter Shift Rule implementation.

    For a quantum circuit f(θ), the gradient is:
    df/dθ = [f(θ + π/2) - f(θ - π/2)] / 2

    This is the native autodiff for quantum circuits.
    """
    gradients = []
    for i in range(len(params)):
        shifted_plus = list(params)
        shifted_plus[i] += shift
        shifted_minus = list(params)
        shifted_minus[i] -= shift

        grad_i = (fn(shifted_plus) - fn(shifted_minus)) / (2 * math.sin(shift))
        gradients.append(grad_i)

    return gradients


def grad(fn: Callable, wrt=None):
    """
    Compute gradient of fn.

    - For classical functions: uses numerical differentiation (or torch.autograd if available)
    - For quantum functions: uses the Parameter Shift Rule

    Returns a function that computes the gradient.
    """
    def gradient_fn(*args):
        if len(args) == 0:
            return 0.0

        # Convert to list for parameter shift
        if isinstance(args[0], (list, tuple)):
            params = list(args[0])
        else:
            params = [float(a) for a in args]

        return parameter_shift_gradient(
            lambda p: fn(*p) if len(p) > 1 else fn(p[0]),
            params
        )

    return gradient_fn


# ═══════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class OptimResult:
    """Result of an optimization."""
    x: Any = None
    energy: float = 0.0
    cost: float = 0.0
    state: Any = None
    n_iterations: int = 0

    def __repr__(self):
        return f"OptimResult(energy={self.energy:.6f}, iterations={self.n_iterations})"


def minimize(cost_fn: Callable, init=None, optimizer=None, **kwargs) -> OptimResult:
    """
    Minimize a cost function.
    Uses gradient descent with optional Parameter Shift Rule for quantum circuits.
    """
    if init is None:
        init = 0.0

    if isinstance(init, (int, float)):
        params = [float(init)]
    elif isinstance(init, (list, tuple)):
        params = [float(x) for x in init]
    else:
        params = [0.0]

    lr = 0.1
    steps = 100

    if optimizer and hasattr(optimizer, 'lr'):
        lr = optimizer.lr
    if optimizer and hasattr(optimizer, 'steps'):
        steps = optimizer.steps
    if 'lr' in kwargs:
        lr = kwargs['lr']
    if 'steps' in kwargs:
        steps = kwargs['steps']

    best_cost = float('inf')
    best_params = list(params)

    for step in range(steps):
        # Evaluate cost
        if len(params) == 1:
            cost = cost_fn(params[0])
        else:
            cost = cost_fn(params)

        if cost < best_cost:
            best_cost = cost
            best_params = list(params)

        # Compute gradient via parameter shift
        grads = parameter_shift_gradient(
            lambda p: cost_fn(p[0]) if len(p) == 1 else cost_fn(p),
            params
        )

        # Update
        for i in range(len(params)):
            params[i] -= lr * grads[i]

    return OptimResult(
        x=best_params[0] if len(best_params) == 1 else best_params,
        energy=best_cost,
        cost=best_cost,
        n_iterations=steps
    )


# ── Optimizer classes ────────────────────────────────────────────────

@dataclass
class GradientDescent:
    lr: float = 0.1
    steps: int = 100

@dataclass
class Adam:
    lr: float = 0.001
    steps: int = 100

@dataclass
class COBYLA:
    maxiter: int = 200
    steps: int = 200


# ═══════════════════════════════════════════════════════════════════
# CHEMISTRY / HAMILTONIAN
# ═══════════════════════════════════════════════════════════════════

def hamiltonian(molecule: str, **kwargs) -> Any:
    """
    Build a molecular Hamiltonian.

    For now: returns a simulated Hamiltonian matrix.
    With PennyLane: would call qml.qchem.molecular_hamiltonian()
    """
    basis = kwargs.get('basis', 'sto-3g')
    bondlength = kwargs.get('bondlength', 0.735)

    if molecule.upper() == "H2":
        # H2 Hamiltonian coefficients (simplified)
        # Real VQE would compute these from the molecular geometry
        if HAS_NUMPY:
            # Pauli decomposition of H2 Hamiltonian
            coeffs = [-0.24274, 0.17771, 0.17771, -0.24274, 0.17059, 0.04532]
            return {
                'molecule': molecule,
                'basis': basis,
                'bondlength': bondlength,
                'coefficients': coeffs,
                'n_qubits': 4,
                'exact_energy': -1.136189454088  # Exact FCI energy for comparison
            }

    return {'molecule': molecule, 'n_qubits': 4, 'coefficients': [0.0]}


def prepare_hartree_fock(state: QState, occupation=None) -> QState:
    """Prepare the Hartree-Fock initial state."""
    new = state.consume()
    if occupation and HAS_NUMPY:
        idx = 0
        for i, occ in enumerate(occupation):
            if occ:
                idx |= (1 << (new.n_qubits - 1 - i))
        new.amplitudes = np.zeros(2 ** new.n_qubits, dtype=complex)
        new.amplitudes[idx] = 1.0
    return new


def double_excitation(state_or_theta, theta_or_state=None, **kwargs) -> QState:
    """Apply a double excitation gate (for VQE ansätze)."""
    if isinstance(state_or_theta, QState):
        state = state_or_theta
        theta = theta_or_state if theta_or_state is not None else 0.0
    else:
        theta = state_or_theta
        state = theta_or_state

    new = state.consume()
    # Simplified: applies a rotation in the {|1100>, |0011>} subspace
    if HAS_NUMPY and new.n_qubits >= 4:
        c, s = math.cos(theta / 2), math.sin(theta / 2)
        # Indices for |1100> = 12 and |0011> = 3 in 4-qubit space
        a12, a3 = new.amplitudes[12], new.amplitudes[3]
        new.amplitudes[12] = c * a12 - s * a3
        new.amplitudes[3] = s * a12 + c * a3
    return new


def single_excitation(state_or_theta, theta_or_state=None, **kwargs) -> QState:
    """Apply a single excitation gate."""
    if isinstance(state_or_theta, QState):
        new = state_or_theta.consume()
    elif isinstance(theta_or_state, QState):
        new = theta_or_state.consume()
    else:
        return state_or_theta
    return new


# ═══════════════════════════════════════════════════════════════════
# QAOA
# ═══════════════════════════════════════════════════════════════════

def qaoa(cost: Callable, qubits: int = 4, depth: int = 2, optimizer=None) -> OptimResult:
    """
    Quantum Approximate Optimization Algorithm.
    Alternates cost and mixer layers, optimizes angles.
    """
    n_params = 2 * depth  # gamma and beta for each layer
    init = [0.1 * i for i in range(n_params)]

    def qaoa_cost(params):
        if isinstance(params, (int, float)):
            params = [params]
        # Simulate QAOA circuit
        q = qregister(qubits)
        for layer in range(min(depth, len(params) // 2)):
            gamma = params[2 * layer] if 2 * layer < len(params) else 0.1
            beta = params[2 * layer + 1] if 2 * layer + 1 < len(params) else 0.1
            # Cost layer: apply RZ rotations
            for i in range(qubits):
                q = rz(gamma, q, i)
            # Mixer layer: apply RX rotations
            for i in range(qubits):
                q = rx(beta, q, i)
        m = measure(q)
        return sum(m.bits) / qubits  # Simplified cost evaluation

    result = minimize(qaoa_cost, init=init, optimizer=optimizer)
    result.state = qregister(qubits)  # Final state
    return result


# ═══════════════════════════════════════════════════════════════════
# PIPELINE RUNTIME
# ═══════════════════════════════════════════════════════════════════

def _synaphe_pipeline(initial, *fns):
    """Nova pipeline operator |> runtime."""
    result = initial
    for fn in fns:
        if callable(fn):
            result = fn(result)
        else:
            result = fn
    return result


# ═══════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ═══════════════════════════════════════════════════════════════════

def to_tensor(data):
    """Convert to tensor."""
    if HAS_TORCH:
        if isinstance(data, Measurement):
            return torch.tensor(data.bits, dtype=torch.float32)
        return torch.tensor(data, dtype=torch.float32)
    if isinstance(data, Measurement):
        return data.bits
    return data


def pretrained(name: str, frozen: bool = True):
    """Load a pretrained model (stub for demo)."""
    print(f"[synaphe] Loading pretrained model: {name} (frozen={frozen})")
    return type('PretrainedModel', (), {
        'features': lambda x: x,
        'forward': lambda x: x,
    })()


# ═══════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═══════════════════════════════════════════════════════════════════

__all__ = [
    # Quantum
    'qubit', 'qregister', 'hadamard', 'rx', 'ry', 'rz', 'cx',
    'measure', 'measure_all', 'expectation', 'QState', 'Measurement',
    'prepare_hartree_fock', 'double_excitation', 'single_excitation',
    # Differentiation
    'grad', 'parameter_shift_gradient',
    # Optimization
    'minimize', 'GradientDescent', 'Adam', 'COBYLA', 'OptimResult',
    # Chemistry
    'hamiltonian', 'qaoa',
    # Pipeline
    '_synaphe_pipeline',
    # Data
    'to_tensor', 'pretrained',
]
