"""
Synaphe Type System v0.3.0

Three pillars:
1. Tensor types with compile-time shape verification
2. Linear types for quantum safety (no-cloning, no-reuse-after-measure)
3. Differentiable types with native parameter shift rule support

This is the "lock" — the type checker (typechecker.py) is the "key."
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Dict, Set, Tuple
from enum import Enum, auto


# ── Kind System ───────────────────────────────────────────────────────
# Every Nova type has a "kind" that determines what operations are legal.

class TypeKind(Enum):
    CLASSICAL = auto()     # Can be freely copied, shared, discarded
    LINEAR = auto()         # Must be used exactly once (quantum states)
    AFFINE = auto()         # Can be discarded but not copied (measured qubits)
    DIFFERENTIABLE = auto() # Tracks gradient information through computation


# ── Base Types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NovaType:
    """Base class for all Nova types."""
    kind: TypeKind = TypeKind.CLASSICAL

    def is_quantum(self) -> bool:
        return self.kind in (TypeKind.LINEAR, TypeKind.AFFINE)

    def is_differentiable(self) -> bool:
        return self.kind == TypeKind.DIFFERENTIABLE

    def is_classical(self) -> bool:
        return self.kind == TypeKind.CLASSICAL


# ── Primitive Types ───────────────────────────────────────────────────

@dataclass(frozen=True)
class IntType(NovaType):
    kind: TypeKind = TypeKind.CLASSICAL
    bits: int = 64

    def __repr__(self):
        return "Int"


@dataclass(frozen=True)
class FloatType(NovaType):
    kind: TypeKind = TypeKind.CLASSICAL
    bits: int = 32

    def __repr__(self):
        return f"Float{self.bits}"


@dataclass(frozen=True)
class BoolType(NovaType):
    kind: TypeKind = TypeKind.CLASSICAL

    def __repr__(self):
        return "Bool"


@dataclass(frozen=True)
class StringType(NovaType):
    kind: TypeKind = TypeKind.CLASSICAL

    def __repr__(self):
        return "String"


@dataclass(frozen=True)
class NoneType(NovaType):
    kind: TypeKind = TypeKind.CLASSICAL

    def __repr__(self):
        return "None"


# ── Tensor Type (Pillar 1: Shape Safety) ─────────────────────────────

@dataclass(frozen=True)
class DimVar:
    """A symbolic dimension variable, e.g. 'batch' in Tensor<f32, [batch, 784]>"""
    name: str

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class DimLit:
    """A concrete dimension literal, e.g. 784 in Tensor<f32, [batch, 784]>"""
    value: int

    def __repr__(self):
        return str(self.value)


# A dimension is either a concrete integer or a symbolic variable
Dim = Union[DimLit, DimVar]


@dataclass(frozen=True)
class TensorType(NovaType):
    """
    Tensor<dtype, [d1, d2, ...]>

    The shape is tracked at compile time. Shape inference rules:
    - matmul: [a, b] @ [b, c] -> [a, c] (b must match)
    - add/sub: shapes must match or broadcast
    - reshape: product of dims must equal product of new dims
    - pipeline: shape flows through each stage
    """
    kind: TypeKind = TypeKind.CLASSICAL
    dtype: str = "Float32"
    shape: Tuple[Dim, ...] = ()

    def ndim(self) -> int:
        return len(self.shape)

    def is_scalar(self) -> bool:
        return len(self.shape) == 0

    def __repr__(self):
        if not self.shape:
            return f"Tensor<{self.dtype}>"
        dims = ", ".join(str(d) for d in self.shape)
        return f"Tensor<{self.dtype}, [{dims}]>"


# ── Quantum Types (Pillar 2: Linear Safety) ──────────────────────────

class QubitState(Enum):
    """Tracks the lifecycle of a quantum resource."""
    FRESH = auto()       # Just allocated, not yet used
    ACTIVE = auto()      # In superposition, entangled, being operated on
    MEASURED = auto()     # Collapsed — cannot be used as quantum anymore
    CONSUMED = auto()     # Passed to another function — no longer owned here


@dataclass(frozen=True)
class QubitType(NovaType):
    """
    A single qubit. LINEAR type — must be used exactly once.

    The No-Cloning Theorem means:
    - You cannot assign a QState to two variables
    - You cannot pass a QState to two different function calls
    - After measurement, the type changes to Classical

    The type checker tracks QubitState through the program.
    """
    kind: TypeKind = TypeKind.LINEAR

    def __repr__(self):
        return "Qubit"


@dataclass(frozen=True)
class QRegisterType(NovaType):
    """
    QRegister<N> — a register of N qubits. LINEAR type.

    Entanglement tracking:
    - cx(0, 1) entangles qubits 0 and 1
    - Measuring qubit 0 after entanglement affects qubit 1
    - The type checker tracks which qubits are entangled
    """
    kind: TypeKind = TypeKind.LINEAR
    n_qubits: int = 0

    def __repr__(self):
        return f"QRegister<{self.n_qubits}>"


@dataclass(frozen=True)
class QStateType(NovaType):
    """
    QState<N> — the quantum state of N qubits. LINEAR type.

    This is the result of evolving a QRegister through a circuit.
    It can be measured (producing a classical result) or passed
    to further quantum operations, but never copied.
    """
    kind: TypeKind = TypeKind.LINEAR
    n_qubits: int = 0

    def __repr__(self):
        return f"QState<{self.n_qubits}>"


@dataclass(frozen=True)
class MeasurementType(NovaType):
    """
    The result of measuring a quantum state. CLASSICAL type.
    Once measured, data is classical and can be freely copied.
    """
    kind: TypeKind = TypeKind.CLASSICAL
    n_bits: int = 0

    def __repr__(self):
        return f"Measurement<{self.n_bits}>"


# ── Differentiable Types (Pillar 3: Autodiff Bridge) ─────────────────

@dataclass(frozen=True)
class GradType(NovaType):
    """
    Grad<T> — wraps any type to track gradient information.

    For classical tensors: uses standard backprop (PyTorch autograd)
    For quantum states: uses the Parameter Shift Rule automatically

    The Parameter Shift Rule:
        d/dθ <ψ(θ)|H|ψ(θ)> = (1/2)[<ψ(θ+π/2)|H|ψ(θ+π/2)> - <ψ(θ-π/2)|H|ψ(θ-π/2)>]

    When the compiler sees grad(quantum_fn), it automatically generates
    the shifted circuit evaluations. The user never writes this.
    """
    kind: TypeKind = TypeKind.DIFFERENTIABLE
    inner_type: NovaType = None

    def __repr__(self):
        return f"Grad<{self.inner_type}>"


@dataclass(frozen=True)
class DiffParam(NovaType):
    """
    A differentiable parameter — the θ in a rotation gate.
    The type checker ensures these are tracked through the computation
    graph so gradients can be computed.
    """
    kind: TypeKind = TypeKind.DIFFERENTIABLE
    name: str = ""

    def __repr__(self):
        return f"DiffParam({self.name})"


# ── Composite Types ──────────────────────────────────────────────────

@dataclass(frozen=True)
class FunctionSig(NovaType):
    """
    Type signature for a function.

    fn train_step(data: Tensor<f32, [4]>, state: QState<4>) -> (Loss, Grad<QState>)

    The type checker verifies:
    - All linear arguments are consumed exactly once
    - Returned linear types are properly owned by the caller
    - Differentiable parameters flow through correctly
    """
    kind: TypeKind = TypeKind.CLASSICAL
    params: List[Tuple[str, NovaType]] = field(default_factory=list)
    return_type: NovaType = None

    def __repr__(self):
        params_str = ", ".join(f"{n}: {t}" for n, t in self.params)
        return f"fn({params_str}) -> {self.return_type}"


@dataclass(frozen=True)
class TupleType(NovaType):
    """Product type: (Loss, Grad<QState>)"""
    kind: TypeKind = TypeKind.CLASSICAL
    elements: Tuple[NovaType, ...] = ()

    def __repr__(self):
        elems = ", ".join(str(e) for e in self.elements)
        return f"({elems})"


@dataclass(frozen=True)
class ListType(NovaType):
    """List<T> — homogeneous list"""
    kind: TypeKind = TypeKind.CLASSICAL
    element_type: NovaType = None

    def __repr__(self):
        return f"List<{self.element_type}>"


@dataclass(frozen=True)
class StreamType(NovaType):
    """Stream<T> — lazy, potentially infinite sequence"""
    kind: TypeKind = TypeKind.CLASSICAL
    element_type: NovaType = None

    def __repr__(self):
        return f"Stream<{self.element_type}>"


@dataclass(frozen=True)
class UnionVariant(NovaType):
    """A variant in a tagged union: Classical(Tensor) | Quantum(QState)"""
    kind: TypeKind = TypeKind.CLASSICAL
    name: str = ""
    inner_type: Optional[NovaType] = None

    def __repr__(self):
        if self.inner_type:
            return f"{self.name}({self.inner_type})"
        return self.name


@dataclass(frozen=True)
class UnionDef(NovaType):
    """Tagged union type"""
    kind: TypeKind = TypeKind.CLASSICAL
    name: str = ""
    variants: Tuple[UnionVariant, ...] = ()

    def __repr__(self):
        return " | ".join(str(v) for v in self.variants)


# ── Hardware Map Types ───────────────────────────────────────────────

class HardwareTarget(Enum):
    CPU = "cpu"
    GPU = "gpu"
    QPU_IBM = "qpu:ibm"
    QPU_GOOGLE = "qpu:google"
    QPU_IONQ = "qpu:ionq"
    QPU_SIM = "qpu:simulator"
    AUTO = "auto"


@dataclass(frozen=True)
class HardwareMapping:
    """
    Maps Nova types to backend implementations.

    Nova Type         | CPU/GPU Backend    | QPU Backend
    ──────────────────|────────────────────|──────────────────
    Tensor<f32, [..]> | torch.Tensor       | (not applicable)
    QRegister<N>      | (simulated)        | qiskit.QuantumCircuit
    QState<N>         | numpy state vector | pennylane.QNode
    Grad<T>           | torch.autograd     | parameter_shift()
    """
    nova_type: str = ""
    cpu_backend: str = ""
    gpu_backend: str = ""
    qpu_backend: str = ""


# Pre-defined hardware mappings
HARDWARE_MAP: Dict[str, HardwareMapping] = {
    "Tensor": HardwareMapping(
        nova_type="Tensor",
        cpu_backend="torch.Tensor (cpu)",
        gpu_backend="torch.Tensor (cuda)",
        qpu_backend="N/A"
    ),
    "QRegister": HardwareMapping(
        nova_type="QRegister",
        cpu_backend="pennylane.default.qubit (sim)",
        gpu_backend="pennylane.lightning.gpu (sim)",
        qpu_backend="qiskit.IBMBackend / pennylane.ionq"
    ),
    "QState": HardwareMapping(
        nova_type="QState",
        cpu_backend="numpy.ndarray (statevector)",
        gpu_backend="cupy.ndarray (statevector)",
        qpu_backend="pennylane.QNode result"
    ),
    "Grad": HardwareMapping(
        nova_type="Grad",
        cpu_backend="torch.autograd.grad",
        gpu_backend="torch.autograd.grad (cuda)",
        qpu_backend="parameter_shift_rule()"
    ),
}


# ── Shape Algebra ────────────────────────────────────────────────────
# Rules for computing output shapes from input shapes.

class ShapeError(Exception):
    """Raised when tensor shapes are incompatible."""
    pass


def matmul_shape(left: TensorType, right: TensorType) -> TensorType:
    """
    Compute the result shape of left @ right.
    Rules: [..., a, b] @ [..., b, c] -> [..., a, c]
    """
    if left.ndim() < 1 or right.ndim() < 1:
        raise ShapeError(f"Cannot matmul scalars: {left} @ {right}")

    l_shape = list(left.shape)
    r_shape = list(right.shape)

    # Last dim of left must match second-to-last of right
    l_inner = l_shape[-1]
    r_inner = r_shape[-2] if len(r_shape) >= 2 else r_shape[0]

    if isinstance(l_inner, DimLit) and isinstance(r_inner, DimLit):
        if l_inner.value != r_inner.value:
            raise ShapeError(
                f"Shape mismatch in matmul: {left} @ {right} — "
                f"inner dimensions {l_inner} vs {r_inner} do not match"
            )

    # Result shape
    if len(r_shape) >= 2:
        result_shape = tuple(l_shape[:-1] + [r_shape[-1]])
    else:
        result_shape = tuple(l_shape[:-1])

    return TensorType(dtype=left.dtype, shape=result_shape)


def broadcast_shape(left: TensorType, right: TensorType) -> TensorType:
    """
    Compute broadcast result shape for elementwise ops (+, -, *, /).
    """
    l_shape = list(left.shape)
    r_shape = list(right.shape)

    # Pad shorter shape with 1s on the left
    max_ndim = max(len(l_shape), len(r_shape))
    l_shape = [DimLit(1)] * (max_ndim - len(l_shape)) + l_shape
    r_shape = [DimLit(1)] * (max_ndim - len(r_shape)) + r_shape

    result = []
    for l, r in zip(l_shape, r_shape):
        if isinstance(l, DimLit) and isinstance(r, DimLit):
            if l.value == r.value:
                result.append(l)
            elif l.value == 1:
                result.append(r)
            elif r.value == 1:
                result.append(l)
            else:
                raise ShapeError(
                    f"Cannot broadcast shapes {left} and {right}: "
                    f"dimensions {l} and {r} are incompatible"
                )
        else:
            # Symbolic dims — assume compatible, unify later
            result.append(l if not isinstance(l, DimLit) or l.value != 1 else r)

    return TensorType(dtype=left.dtype, shape=tuple(result))


def pipeline_shape(input_type: NovaType, fn_sig: FunctionSig) -> NovaType:
    """
    Compute the output type of piping input_type through fn_sig.
    Verifies that input_type matches fn_sig's first parameter.
    """
    if not fn_sig.params:
        raise ShapeError(f"Cannot pipe into function with no parameters: {fn_sig}")

    expected = fn_sig.params[0][1]

    # Check compatibility
    if isinstance(input_type, TensorType) and isinstance(expected, TensorType):
        if input_type.shape and expected.shape:
            # Verify shapes match (concrete dims only)
            for actual, exp in zip(input_type.shape, expected.shape):
                if isinstance(actual, DimLit) and isinstance(exp, DimLit):
                    if actual.value != exp.value:
                        raise ShapeError(
                            f"Pipeline shape mismatch: got {input_type} "
                            f"but function expects {expected}"
                        )

    return fn_sig.return_type
