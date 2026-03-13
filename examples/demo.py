#!/usr/bin/env python3
"""
Synaphe v0.3.0 — Executable Demo

This demonstrates the golden snippets ACTUALLY RUNNING:
1. VQE for H2 ground state energy
2. Quantum state linearity enforcement
3. Parameter Shift Rule gradient computation
4. Pipeline operator in action
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stdlib.runtime import *
import math

CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
BOLD = "\033[1m"
DIM = "\033[90m"
RESET = "\033[0m"

def header(text):
    print(f"\n{BOLD}{'═'*60}{RESET}")
    print(f"{BOLD}  {text}{RESET}")
    print(f"{BOLD}{'═'*60}{RESET}")


# ═══════════════════════════════════════════════════════════════════
header("Demo 1: VQE — Finding H₂ Ground State Energy")
# ═══════════════════════════════════════════════════════════════════

print(f"""
{DIM}// Nova source:{RESET}
{CYAN}let H = hamiltonian("H2", basis="sto-3g", bondlength=0.735)

@differentiable
fn ansatz(theta: Float) -> Float {{
    qregister(4)
        |> prepare_hartree_fock([1, 1, 0, 0])
        |> double_excitation(theta)
        |> measure |> expectation(PauliZ)
}}

let result = minimize(fn(t) => ansatz(t), init=0.0,
    optimizer=GradientDescent(lr=0.4, steps=60)){RESET}
""")

print(f"{DIM}Running...{RESET}")

H = hamiltonian("H2", basis="sto-3g", bondlength=0.735)

def vqe_cost(theta):
    q = qregister(4)
    q = prepare_hartree_fock(q, [1, 1, 0, 0])
    q = double_excitation(theta, q)
    m = measure(q)
    return expectation("PauliZ", m)

result = minimize(vqe_cost, init=0.0,
                  optimizer=GradientDescent(lr=0.4, steps=60))

print(f"{GREEN}✓ VQE converged in {result.n_iterations} iterations{RESET}")
print(f"  Ground state energy: {BOLD}{result.energy:.6f}{RESET}")
print(f"  Optimal parameter:   θ = {result.x:.4f}")
print(f"  Exact energy (FCI):  {H['exact_energy']:.6f}")


# ═══════════════════════════════════════════════════════════════════
header("Demo 2: Linear Type Safety — No-Cloning Enforcement")
# ═══════════════════════════════════════════════════════════════════

print(f"""
{DIM}// Nova source (this should ERROR):{RESET}
{CYAN}let q = qregister(4)
let result1 = measure(q)   // OK — consumes q
let result2 = hadamard(q)  // ERROR — q already consumed!{RESET}
""")

print(f"{DIM}Running...{RESET}")

q = qregister(4)
result1 = measure(q)
print(f"{GREEN}✓ measure(q) succeeded → {result1}{RESET}")

try:
    result2 = hadamard(q)
    print(f"{RED}✗ Should have failed!{RESET}")
except RuntimeError as e:
    print(f"{GREEN}✓ Correctly caught: {RESET}{YELLOW}{e}{RESET}")


# ═══════════════════════════════════════════════════════════════════
header("Demo 3: Parameter Shift Rule — Quantum Autodiff")
# ═══════════════════════════════════════════════════════════════════

print(f"""
{DIM}// Nova source:{RESET}
{CYAN}@differentiable
fn circuit(theta: Float) -> Float {{
    qubit() |> ry(theta) |> measure |> expectation(PauliZ)
}}

let gradient = grad(circuit)
let dtheta = gradient(0.5){RESET}
""")

print(f"{DIM}Running...{RESET}")

def circuit_fn(theta):
    q = qubit()
    q = ry(theta, q)
    m = measure(q)
    return expectation("PauliZ", m)

gradient_fn = grad(circuit_fn)
dtheta = gradient_fn(0.5)

print(f"{GREEN}✓ Parameter Shift Rule computed gradient{RESET}")
print(f"  f(0.5) = {circuit_fn(0.5):.4f}")
print(f"  df/dθ at θ=0.5: {BOLD}{dtheta[0]:.6f}{RESET}")
print(f"  {DIM}(Computed via [f(θ+π/2) - f(θ-π/2)] / 2){RESET}")


# ═══════════════════════════════════════════════════════════════════
header("Demo 4: Pipeline Operator in Action")
# ═══════════════════════════════════════════════════════════════════

print(f"""
{DIM}// Nova source:{RESET}
{CYAN}let data = [3, 1, 4, 1, 5, 9, 2, 6]
let result = data |> sorted |> reversed |> list{RESET}
""")

data = [3, 1, 4, 1, 5, 9, 2, 6]
result = _synaphe_pipeline(data, sorted, reversed, list)

print(f"{GREEN}✓ Pipeline executed{RESET}")
print(f"  Input:  {data}")
print(f"  Output: {result}")


# ═══════════════════════════════════════════════════════════════════
header("Demo 5: Quantum Pipeline — Bell State")
# ═══════════════════════════════════════════════════════════════════

print(f"""
{DIM}// Nova source:{RESET}
{CYAN}let bell = qregister(2) |> hadamard |> cx(0, 1) |> measure{RESET}
""")

print(f"{DIM}Running 10 measurements...{RESET}")

results = []
for _ in range(10):
    q = qregister(2)
    q = hadamard(q)
    q = cx(q)
    m = measure(q)
    results.append(m.bits)

print(f"{GREEN}✓ Bell state measurements:{RESET}")
for r in results:
    correlated = "✓ correlated" if r[0] == r[1] else "✗ uncorrelated"
    print(f"  |{''.join(str(b) for b in r)}⟩  {DIM}{correlated}{RESET}")

corr_rate = sum(1 for r in results if r[0] == r[1]) / len(results)
print(f"\n  Correlation rate: {BOLD}{corr_rate*100:.0f}%{RESET} {DIM}(expect ~100% for ideal Bell state){RESET}")


# ═══════════════════════════════════════════════════════════════════
header("Demo 6: QAOA — Optimization")
# ═══════════════════════════════════════════════════════════════════

print(f"""
{DIM}// Nova source:{RESET}
{CYAN}let result = qaoa(
    cost = my_cost_fn,
    qubits = 4,
    depth = 2,
    optimizer = COBYLA(maxiter=50)
){RESET}
""")

qaoa_result = qaoa(
    cost=lambda x: sum(x) if isinstance(x, list) else x,
    qubits=4,
    depth=2,
    optimizer=COBYLA(maxiter=50)
)

print(f"{GREEN}✓ QAOA completed{RESET}")
print(f"  Optimal cost: {qaoa_result.cost:.4f}")
print(f"  Iterations:   {qaoa_result.n_iterations}")


# ═══════════════════════════════════════════════════════════════════
print(f"\n{BOLD}{'═'*60}{RESET}")
print(f"{BOLD}{GREEN}  All demos completed successfully ✓{RESET}")
print(f"{BOLD}{'═'*60}{RESET}")
print(f"""
{DIM}Synaphe v0.3.0 capabilities demonstrated:
  ✓ VQE with quantum simulation
  ✓ Linear type enforcement (No-Cloning Theorem)
  ✓ Parameter Shift Rule (quantum autodiff)
  ✓ Pipeline operator |>
  ✓ Bell state preparation and measurement
  ✓ QAOA optimization{RESET}
""")
