#!/usr/bin/env python3
"""
Synaphe v0.3.0 Test Suite — Hardware Constraints & Data Pipelines
The Synaphe Project
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hardware import (
    HardwareChecker, HardwareProfile, HARDWARE_PROFILES,
    CircuitProfile, GateOp, check_fidelity, estimate_fidelity,
    HardwareWarning
)
from stdlib.data import (
    Schema, SchemaField, Dataset,
    generate_synthetic, load_mnist_sample, normalize, batch,
    shuffle, flatten, one_hot, split
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
# HARDWARE CONSTRAINT TESTS
# ═══════════════════════════════════════════════════════════════════

def test_simulator_accepts_anything():
    """Ideal simulator should accept any circuit."""
    hw = HARDWARE_PROFILES["simulator"]
    checker = HardwareChecker(hw)
    circuit = CircuitProfile(n_qubits=10)
    for _ in range(50):
        circuit.add_gate(GateOp("H", [0]), hw)
        circuit.add_gate(GateOp("CNOT", [0, 1]), hw)
    warnings = checker.check_circuit(circuit)
    errors = [w for w in warnings if w.severity == "error"]
    assert len(errors) == 0, f"Simulator should accept anything: {errors}"

def test_too_many_qubits():
    """Circuit with more qubits than hardware should error."""
    hw = HARDWARE_PROFILES["google_sycamore"]  # 53 qubits
    checker = HardwareChecker(hw)
    circuit = CircuitProfile(n_qubits=100)
    warnings = checker.check_circuit(circuit)
    capacity_errors = [w for w in warnings if w.category == "capacity"]
    assert len(capacity_errors) > 0, "Should flag qubit count exceeded"

def test_circuit_too_deep():
    """Very deep circuit should trigger depth warning/error."""
    hw = HARDWARE_PROFILES["google_sycamore"]  # max_depth=200
    checker = HardwareChecker(hw)
    circuit = CircuitProfile(n_qubits=4)
    circuit.depth = 500  # Way too deep
    warnings = checker.check_circuit(circuit)
    depth_issues = [w for w in warnings if w.category == "depth"]
    assert len(depth_issues) > 0, "Should flag circuit too deep"

def test_coherence_exceeded():
    """Circuit longer than T2 should error."""
    hw = HARDWARE_PROFILES["google_sycamore"]  # T2 = 10 μs
    checker = HardwareChecker(hw)
    circuit = CircuitProfile(n_qubits=4)
    # Add enough gates to exceed 10 μs (10,000 ns)
    for _ in range(400):
        circuit.add_gate(GateOp("CNOT", [0, 1]), hw)  # ~44ns each = 17,600 ns
    warnings = checker.check_circuit(circuit)
    coherence_issues = [w for w in warnings if w.category == "coherence"]
    assert len(coherence_issues) > 0, "Should flag coherence time exceeded"

def test_ionq_long_coherence():
    """IonQ's long coherence time should accept deep circuits."""
    hw = HARDWARE_PROFILES["ionq_forte"]  # T2 = 1,000,000 μs
    checker = HardwareChecker(hw)
    circuit = CircuitProfile(n_qubits=4)
    for _ in range(100):
        circuit.add_gate(GateOp("H", [0]), hw)
        circuit.add_gate(GateOp("MS", [0, 1]), hw)
    warnings = checker.check_circuit(circuit)
    coherence_errors = [w for w in warnings
                        if w.category == "coherence" and w.severity == "error"]
    assert len(coherence_errors) == 0, "IonQ should handle deep circuits"

def test_fidelity_estimation():
    """Circuit fidelity should decrease with more gates."""
    fid_10 = estimate_fidelity(4, 10, 2, "ibm_brisbane")
    fid_100 = estimate_fidelity(4, 100, 20, "ibm_brisbane")
    fid_500 = estimate_fidelity(4, 500, 100, "ibm_brisbane")
    assert fid_10 > fid_100 > fid_500, (
        f"Fidelity should decrease: {fid_10:.3f} > {fid_100:.3f} > {fid_500:.3f}"
    )

def test_connectivity_warning():
    """Non-adjacent qubits on IBM should trigger connectivity warning."""
    hw = HARDWARE_PROFILES["ibm_brisbane"]
    checker = HardwareChecker(hw)
    circuit = CircuitProfile(n_qubits=20)
    # Qubits 0 and 15 are NOT directly connected
    circuit.add_gate(GateOp("CNOT", [0, 15]), hw)
    warnings = checker.check_circuit(circuit)
    conn_issues = [w for w in warnings if w.category == "connectivity"]
    assert len(conn_issues) > 0, "Should flag non-adjacent qubits"

def test_ionq_all_to_all():
    """IonQ has all-to-all connectivity — no warnings for any qubit pair."""
    hw = HARDWARE_PROFILES["ionq_forte"]
    checker = HardwareChecker(hw)
    circuit = CircuitProfile(n_qubits=10)
    circuit.add_gate(GateOp("MS", [0, 9]), hw)
    circuit.add_gate(GateOp("MS", [3, 7]), hw)
    warnings = checker.check_circuit(circuit)
    conn_issues = [w for w in warnings if w.category == "connectivity"]
    assert len(conn_issues) == 0, "IonQ should allow any qubit pair"

def test_check_fidelity_api():
    """The user-facing check_fidelity function should work."""
    warnings = check_fidelity(
        n_qubits=4,
        gate_ops=[
            {"name": "H", "qubits": [0]},
            {"name": "CNOT", "qubits": [0, 1]},
            {"name": "H", "qubits": [1]},
            {"name": "CNOT", "qubits": [1, 2]},
        ],
        target="ibm_brisbane"
    )
    assert len(warnings) > 0, "Should produce at least fidelity info"

def test_hardware_profiles_exist():
    """All named profiles should be loadable."""
    for name in ["simulator", "ibm_brisbane", "ibm_sherbrooke",
                 "google_sycamore", "ionq_forte"]:
        hw = HARDWARE_PROFILES[name]
        assert hw.n_qubits > 0, f"{name} should have qubits"
        assert hw.t1_us > 0, f"{name} should have T1 time"


# ═══════════════════════════════════════════════════════════════════
# DATA PIPELINE TESTS
# ═══════════════════════════════════════════════════════════════════

def test_synthetic_dataset():
    """Generate synthetic classification data."""
    ds = generate_synthetic(n_samples=200, n_features=8, n_classes=3)
    assert ds.n_samples == 200
    assert ds.data is not None

def test_mnist_sample():
    """Generate MNIST-like sample data."""
    ds = load_mnist_sample(n=50)
    assert ds.n_samples == 50

def test_normalize():
    """Normalize should center and scale data."""
    import numpy as np
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    normed = normalize(data)
    assert abs(np.mean(normed)) < 0.01, "Mean should be ~0"
    assert abs(np.std(normed) - 1.0) < 0.1, "Std should be ~1"

def test_batch():
    """Batch should split data into chunks."""
    data = list(range(100))
    batches = batch(data, size=32)
    assert len(batches) == 4  # 32+32+32+4
    assert len(batches[0]) == 32
    assert len(batches[-1]) == 4

def test_shuffle():
    """Shuffle should randomize order."""
    data = list(range(100))
    shuffled = shuffle(data, seed=42)
    assert len(shuffled) == 100
    assert shuffled != data  # Extremely unlikely to be identical

def test_one_hot():
    """One-hot encoding should work."""
    labels = [0, 1, 2, 1, 0]
    encoded = one_hot(labels, n_classes=3)
    import numpy as np
    encoded = np.array(encoded)
    assert encoded.shape == (5, 3)
    assert encoded[0, 0] == 1.0
    assert encoded[1, 1] == 1.0
    assert encoded[2, 2] == 1.0

def test_split():
    """Split should divide dataset into train/test."""
    ds = generate_synthetic(n_samples=100)
    train, test = split(ds, ratio=0.8)
    assert train.n_samples == 80
    assert test.n_samples == 20

def test_schema_validation_pass():
    """Valid record should pass schema validation."""
    schema = Schema(name="UserInput", fields=[
        SchemaField("age", "int", constraints=[lambda x: x >= 0]),
        SchemaField("name", "string"),
        SchemaField("score", "float", constraints=[lambda x: x >= 0.0]),
    ])
    valid, errors = schema.validate({"age": 25, "name": "Alice", "score": 95.5})
    assert valid, f"Should pass: {errors}"

def test_schema_validation_fail():
    """Invalid record should fail schema validation."""
    schema = Schema(name="UserInput", fields=[
        SchemaField("age", "int", constraints=[lambda x: x >= 0]),
        SchemaField("name", "string"),
    ])
    valid, errors = schema.validate({"age": -5, "name": 42})
    assert not valid, "Should fail: wrong types and constraint violated"

def test_pipeline_data_flow():
    """Data should flow through a pipeline of transforms."""
    ds = generate_synthetic(n_samples=100, n_features=4)
    import numpy as np
    data = np.array(ds.data) if not isinstance(ds.data, np.ndarray) else ds.data
    result = batch(normalize(shuffle(data, seed=42)), size=25)
    assert len(result) == 4
    assert abs(np.mean(result[0])) < 1.0


# ═══════════════════════════════════════════════════════════════════
# RUN ALL TESTS
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\n\033[1m🔬 Synaphe v0.3.0 — The Synaphe Project\033[0m")
    print(f"\033[1m   Hardware Constraints & Data Pipeline Tests\033[0m\n")

    print("\033[1mHardware Constraint Checks:\033[0m")
    test("simulator accepts any circuit", test_simulator_accepts_anything)
    test("too many qubits → ERROR", test_too_many_qubits)
    test("circuit too deep → WARNING", test_circuit_too_deep)
    test("coherence time exceeded → ERROR", test_coherence_exceeded)
    test("IonQ long coherence handles deep circuits", test_ionq_long_coherence)
    test("fidelity decreases with gate count", test_fidelity_estimation)
    test("non-adjacent qubits → connectivity WARNING", test_connectivity_warning)
    test("IonQ all-to-all → no connectivity warnings", test_ionq_all_to_all)
    test("check_fidelity() user API works", test_check_fidelity_api)
    test("all hardware profiles loadable", test_hardware_profiles_exist)

    print(f"\n\033[1mData Pipeline:\033[0m")
    test("generate synthetic dataset", test_synthetic_dataset)
    test("generate MNIST sample", test_mnist_sample)
    test("normalize to mean=0, std=1", test_normalize)
    test("batch into chunks", test_batch)
    test("shuffle randomizes order", test_shuffle)
    test("one-hot encoding", test_one_hot)
    test("train/test split", test_split)
    test("schema validation passes valid data", test_schema_validation_pass)
    test("schema validation catches invalid data", test_schema_validation_fail)
    test("pipeline: shuffle |> normalize |> batch", test_pipeline_data_flow)

    print(f"\n{'='*55}")
    print(f"\033[1mResults: {tests_passed}/{tests_run} passed\033[0m", end="")
    if tests_passed == tests_run:
        print(f" \033[32m— ALL PASS ✓\033[0m")
    else:
        print(f" \033[31m— {tests_run - tests_passed} FAILED\033[0m")
    print()
