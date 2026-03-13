"""
Synaphe Hardware Constraint System v0.3.0
The Synaphe Project

Real QPUs are noisy. This module adds hardware-aware type checking:

1. COHERENCE CHECKING: Is the circuit too deep for the hardware's T1/T2 time?
2. GATE FIDELITY: Will accumulated gate errors make the result meaningless?
3. CONNECTIVITY: Can these two qubits actually interact on the target topology?
4. NOISE MODELING: @noise_model decorator for realistic simulation.

The type checker warns BEFORE you waste $200 of QPU time on a garbage result.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum, auto
import math


# ═══════════════════════════════════════════════════════════════════
# HARDWARE PROFILES
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class GateSpec:
    """Specification for a single gate type on hardware."""
    name: str
    fidelity: float          # 0.0 to 1.0 (1.0 = perfect)
    duration_ns: float       # Gate duration in nanoseconds
    native: bool = True      # Is this a native gate? (non-native = decomposed)


@dataclass
class HardwareProfile:
    """
    Complete specification of a quantum hardware backend.
    Used by the type checker to validate circuits before execution.
    """
    name: str
    provider: str
    n_qubits: int
    t1_us: float                    # T1 relaxation time (microseconds)
    t2_us: float                    # T2 dephasing time (microseconds)
    gate_specs: Dict[str, GateSpec] = field(default_factory=dict)
    connectivity: Set[Tuple[int, int]] = field(default_factory=set)
    readout_fidelity: float = 0.95  # Measurement fidelity
    max_circuit_depth: int = 1000   # Hardware limit on circuit depth
    shot_time_us: float = 1.0      # Time per measurement shot

    @property
    def t1_ns(self) -> float:
        return self.t1_us * 1000

    @property
    def t2_ns(self) -> float:
        return self.t2_us * 1000

    def are_connected(self, q1: int, q2: int) -> bool:
        """Check if two qubits can directly interact."""
        if not self.connectivity:
            return True  # No connectivity constraints (simulator)
        return (q1, q2) in self.connectivity or (q2, q1) in self.connectivity

    def gate_fidelity(self, gate_name: str) -> float:
        """Get fidelity for a specific gate."""
        spec = self.gate_specs.get(gate_name)
        return spec.fidelity if spec else 0.99

    def gate_duration(self, gate_name: str) -> float:
        """Get duration in ns for a specific gate."""
        spec = self.gate_specs.get(gate_name)
        return spec.duration_ns if spec else 50.0


# ── Pre-defined Hardware Profiles ────────────────────────────────────

def _ibm_eagle_connectivity(n: int = 127) -> Set[Tuple[int, int]]:
    """Heavy-hex lattice connectivity (simplified)."""
    edges = set()
    for i in range(min(n - 1, 20)):
        edges.add((i, i + 1))
        if i + 4 < n:
            edges.add((i, i + 4))
    return edges


HARDWARE_PROFILES: Dict[str, HardwareProfile] = {
    "simulator": HardwareProfile(
        name="Ideal Simulator",
        provider="synaphe",
        n_qubits=32,
        t1_us=float('inf'),
        t2_us=float('inf'),
        gate_specs={
            "H": GateSpec("H", 1.0, 0.0),
            "RX": GateSpec("RX", 1.0, 0.0),
            "RY": GateSpec("RY", 1.0, 0.0),
            "RZ": GateSpec("RZ", 1.0, 0.0),
            "CNOT": GateSpec("CNOT", 1.0, 0.0),
            "CZ": GateSpec("CZ", 1.0, 0.0),
        },
        readout_fidelity=1.0,
        max_circuit_depth=100000,
    ),

    "ibm_brisbane": HardwareProfile(
        name="IBM Brisbane (Eagle r3)",
        provider="ibm",
        n_qubits=127,
        t1_us=250.0,         # ~250 μs typical
        t2_us=150.0,         # ~150 μs typical
        gate_specs={
            "RZ": GateSpec("RZ", 0.9998, 0.0, native=True),    # Virtual gate
            "SX": GateSpec("SX", 0.9996, 25.0, native=True),   # √X
            "X": GateSpec("X", 0.9996, 25.0, native=True),
            "CNOT": GateSpec("CNOT", 0.990, 300.0, native=True),  # ECR-based
            "H": GateSpec("H", 0.9994, 50.0, native=False),    # Decomposed to SX+RZ
            "RX": GateSpec("RX", 0.9994, 50.0, native=False),
            "RY": GateSpec("RY", 0.9994, 50.0, native=False),
        },
        connectivity=_ibm_eagle_connectivity(127),
        readout_fidelity=0.97,
        max_circuit_depth=300,
    ),

    "ibm_sherbrooke": HardwareProfile(
        name="IBM Sherbrooke (Eagle r3)",
        provider="ibm",
        n_qubits=127,
        t1_us=300.0,
        t2_us=200.0,
        gate_specs={
            "RZ": GateSpec("RZ", 0.9999, 0.0, native=True),
            "SX": GateSpec("SX", 0.9997, 22.0, native=True),
            "X": GateSpec("X", 0.9997, 22.0, native=True),
            "CNOT": GateSpec("CNOT", 0.993, 270.0, native=True),
            "H": GateSpec("H", 0.9995, 44.0, native=False),
            "RX": GateSpec("RX", 0.9995, 44.0, native=False),
            "RY": GateSpec("RY", 0.9995, 44.0, native=False),
        },
        connectivity=_ibm_eagle_connectivity(127),
        readout_fidelity=0.98,
        max_circuit_depth=300,
    ),

    "google_sycamore": HardwareProfile(
        name="Google Sycamore",
        provider="google",
        n_qubits=53,
        t1_us=20.0,
        t2_us=10.0,
        gate_specs={
            "SYC": GateSpec("SYC", 0.995, 12.0, native=True),   # Sycamore gate
            "PHASED_XZ": GateSpec("PHASED_XZ", 0.999, 25.0, native=True),
            "CZ": GateSpec("CZ", 0.993, 32.0, native=True),
            "H": GateSpec("H", 0.999, 25.0, native=False),
            "RX": GateSpec("RX", 0.999, 25.0, native=False),
            "RY": GateSpec("RY", 0.999, 25.0, native=False),
            "RZ": GateSpec("RZ", 0.9999, 0.0, native=True),
            "CNOT": GateSpec("CNOT", 0.993, 44.0, native=False),
        },
        readout_fidelity=0.96,
        max_circuit_depth=200,
    ),

    "ionq_forte": HardwareProfile(
        name="IonQ Forte (Trapped Ion)",
        provider="ionq",
        n_qubits=36,
        t1_us=10_000_000.0,   # ~10 seconds! Ions have incredible coherence
        t2_us=1_000_000.0,    # ~1 second
        gate_specs={
            "GPI": GateSpec("GPI", 0.9998, 135.0, native=True),
            "GPI2": GateSpec("GPI2", 0.9998, 135.0, native=True),
            "MS": GateSpec("MS", 0.985, 600.0, native=True),    # Mølmer-Sørensen
            "H": GateSpec("H", 0.9997, 135.0, native=False),
            "RX": GateSpec("RX", 0.9997, 135.0, native=False),
            "RY": GateSpec("RY", 0.9997, 135.0, native=False),
            "RZ": GateSpec("RZ", 0.9999, 0.0, native=True),
            "CNOT": GateSpec("CNOT", 0.985, 600.0, native=False),
        },
        connectivity=set(),  # All-to-all connectivity!
        readout_fidelity=0.995,
        max_circuit_depth=500,
    ),
}


# ═══════════════════════════════════════════════════════════════════
# CIRCUIT ANALYSIS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class GateOp:
    """A single gate operation in a circuit."""
    name: str
    qubits: List[int]
    params: List[float] = field(default_factory=list)
    line: int = 0
    col: int = 0


@dataclass
class CircuitProfile:
    """Analysis of a quantum circuit's resource requirements."""
    n_qubits: int = 0
    depth: int = 0
    gate_count: int = 0
    two_qubit_count: int = 0
    gates: List[GateOp] = field(default_factory=list)
    total_duration_ns: float = 0.0

    def add_gate(self, op: GateOp, hw: HardwareProfile):
        self.gates.append(op)
        self.gate_count += 1
        if len(op.qubits) >= 2:
            self.two_qubit_count += 1
        self.total_duration_ns += hw.gate_duration(op.name)
        # Simplified depth tracking
        self.depth = max(self.depth, self.gate_count // max(self.n_qubits, 1) + 1)


# ═══════════════════════════════════════════════════════════════════
# HARDWARE-AWARE CHECKER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class HardwareWarning:
    """A warning about hardware constraints."""
    severity: str  # "info", "warning", "error"
    category: str  # "coherence", "fidelity", "connectivity", "depth"
    message: str
    line: int = 0

    def __repr__(self):
        icon = {"info": "ℹ", "warning": "⚠", "error": "✗"}[self.severity]
        return f"{icon} [{self.category}] {self.message}"


class HardwareChecker:
    """
    Validates quantum circuits against hardware constraints.

    Checks:
    1. Circuit depth vs coherence time
    2. Accumulated gate infidelity
    3. Qubit connectivity (can these qubits interact?)
    4. Gate set support (is this gate native?)
    5. Qubit count vs hardware capacity
    """

    def __init__(self, profile: HardwareProfile):
        self.hw = profile
        self.warnings: List[HardwareWarning] = []

    def check_circuit(self, circuit: CircuitProfile) -> List[HardwareWarning]:
        """Run all hardware checks on a circuit profile."""
        self.warnings = []

        self._check_qubit_count(circuit)
        self._check_depth(circuit)
        self._check_coherence(circuit)
        self._check_fidelity(circuit)
        self._check_connectivity(circuit)
        self._check_native_gates(circuit)

        return self.warnings

    def _check_qubit_count(self, circuit: CircuitProfile):
        if circuit.n_qubits > self.hw.n_qubits:
            self.warnings.append(HardwareWarning(
                severity="error",
                category="capacity",
                message=(
                    f"Circuit requires {circuit.n_qubits} qubits but "
                    f"{self.hw.name} only has {self.hw.n_qubits}. "
                    f"Consider circuit cutting or a larger device."
                )
            ))

    def _check_depth(self, circuit: CircuitProfile):
        if circuit.depth > self.hw.max_circuit_depth:
            self.warnings.append(HardwareWarning(
                severity="error",
                category="depth",
                message=(
                    f"Circuit depth {circuit.depth} exceeds {self.hw.name}'s "
                    f"maximum of {self.hw.max_circuit_depth}. "
                    f"Reduce depth or use circuit optimization passes."
                )
            ))
        elif circuit.depth > self.hw.max_circuit_depth * 0.7:
            self.warnings.append(HardwareWarning(
                severity="warning",
                category="depth",
                message=(
                    f"Circuit depth {circuit.depth} is {circuit.depth/self.hw.max_circuit_depth*100:.0f}% "
                    f"of {self.hw.name}'s limit ({self.hw.max_circuit_depth}). "
                    f"Results may be noisy."
                )
            ))

    def _check_coherence(self, circuit: CircuitProfile):
        """Check if circuit duration exceeds coherence time."""
        duration_us = circuit.total_duration_ns / 1000

        if duration_us > self.hw.t2_us:
            self.warnings.append(HardwareWarning(
                severity="error",
                category="coherence",
                message=(
                    f"Circuit duration ({duration_us:.1f} μs) exceeds "
                    f"{self.hw.name}'s T2 coherence time ({self.hw.t2_us:.0f} μs). "
                    f"Qubits will decohere before the circuit completes — "
                    f"results will be random noise."
                )
            ))
        elif duration_us > self.hw.t2_us * 0.5:
            self.warnings.append(HardwareWarning(
                severity="warning",
                category="coherence",
                message=(
                    f"Circuit duration ({duration_us:.1f} μs) is "
                    f"{duration_us/self.hw.t2_us*100:.0f}% of T2 time "
                    f"({self.hw.t2_us:.0f} μs). Expect significant decoherence noise."
                )
            ))

    def _check_fidelity(self, circuit: CircuitProfile):
        """Estimate overall circuit fidelity from gate error accumulation."""
        total_fidelity = 1.0

        for gate in circuit.gates:
            gate_fid = self.hw.gate_fidelity(gate.name)
            total_fidelity *= gate_fid

        # Include readout fidelity for each measured qubit
        total_fidelity *= self.hw.readout_fidelity ** circuit.n_qubits

        if total_fidelity < 0.10:
            self.warnings.append(HardwareWarning(
                severity="error",
                category="fidelity",
                message=(
                    f"Estimated circuit fidelity is {total_fidelity*100:.1f}% — "
                    f"below 10%. Results will be dominated by noise. "
                    f"Reduce gate count (currently {circuit.gate_count}) or use "
                    f"error mitigation techniques."
                )
            ))
        elif total_fidelity < 0.50:
            self.warnings.append(HardwareWarning(
                severity="warning",
                category="fidelity",
                message=(
                    f"Estimated circuit fidelity is {total_fidelity*100:.1f}% "
                    f"({circuit.gate_count} gates, {circuit.two_qubit_count} two-qubit). "
                    f"Consider error mitigation or circuit optimization."
                )
            ))
        else:
            self.warnings.append(HardwareWarning(
                severity="info",
                category="fidelity",
                message=(
                    f"Estimated circuit fidelity: {total_fidelity*100:.1f}% "
                    f"({circuit.gate_count} gates). Acceptable for NISQ execution."
                )
            ))

    def _check_connectivity(self, circuit: CircuitProfile):
        """Check that all two-qubit gates respect hardware topology."""
        if not self.hw.connectivity:
            return  # All-to-all connectivity (IonQ, simulator)

        for gate in circuit.gates:
            if len(gate.qubits) >= 2:
                q1, q2 = gate.qubits[0], gate.qubits[1]
                if not self.hw.are_connected(q1, q2):
                    self.warnings.append(HardwareWarning(
                        severity="warning",
                        category="connectivity",
                        message=(
                            f"L{gate.line}: {gate.name}({q1}, {q2}) — qubits {q1} and {q2} "
                            f"are not directly connected on {self.hw.name}. "
                            f"SWAP gates will be inserted (adds ~3x depth)."
                        )
                    ))

    def _check_native_gates(self, circuit: CircuitProfile):
        """Check if non-native gates need decomposition."""
        non_native = set()
        for gate in circuit.gates:
            spec = self.hw.gate_specs.get(gate.name)
            if spec and not spec.native:
                non_native.add(gate.name)

        if non_native:
            self.warnings.append(HardwareWarning(
                severity="info",
                category="gates",
                message=(
                    f"Gates {non_native} are not native to {self.hw.name} "
                    f"and will be decomposed. This increases circuit depth."
                )
            ))


# ═══════════════════════════════════════════════════════════════════
# CHECK FIDELITY — The user-facing function
# ═══════════════════════════════════════════════════════════════════

def check_fidelity(n_qubits: int, gate_ops: List[dict],
                   target: str = "ibm_brisbane") -> List[HardwareWarning]:
    """
    Check if a circuit can run faithfully on target hardware.

    Usage in Nova:
        check_fidelity(my_circuit, target="ibm_brisbane")

    Returns a list of warnings/errors about hardware constraints.
    """
    hw = HARDWARE_PROFILES.get(target, HARDWARE_PROFILES["simulator"])
    checker = HardwareChecker(hw)

    circuit = CircuitProfile(n_qubits=n_qubits)
    for op in gate_ops:
        gate = GateOp(
            name=op.get("name", "H"),
            qubits=op.get("qubits", [0]),
            params=op.get("params", []),
            line=op.get("line", 0)
        )
        circuit.add_gate(gate, hw)

    return checker.check_circuit(circuit)


def estimate_fidelity(n_qubits: int, n_gates: int, n_two_qubit: int,
                      target: str = "ibm_brisbane") -> float:
    """Quick estimate of circuit fidelity without full circuit analysis."""
    hw = HARDWARE_PROFILES.get(target, HARDWARE_PROFILES["simulator"])

    single_q_fid = hw.gate_fidelity("H")
    two_q_fid = hw.gate_fidelity("CNOT")
    n_single = n_gates - n_two_qubit

    fidelity = (single_q_fid ** n_single) * (two_q_fid ** n_two_qubit)
    fidelity *= hw.readout_fidelity ** n_qubits

    return fidelity
