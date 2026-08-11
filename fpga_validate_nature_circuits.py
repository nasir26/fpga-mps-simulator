"""
Push circuits produced by the Qiskit Nature tutorial notebooks (this folder's
01-12 .ipynb files) through the real FPGA MPS simulator (FPGAMPSSimulator,
Alveo U55C hardware) and cross-check against the exact/Aer reference values
the notebooks themselves report.

Three circuits, all reconstructed with the *exact* code from the notebooks:

  1. 11_quadratic_hamiltonian_and_slater_determinants.ipynb
     -> FermionicGaussianState eigenstate-prep circuit (4 qubits)
  2. 11_quadratic_hamiltonian_and_slater_determinants.ipynb
     -> SlaterDeterminant eigenstate-prep circuit (4 qubits)
  3. 03_ground_state_solvers.ipynb
     -> UCCSD+HartreeFock VQE ground-state circuit for H2/STO-3G (4 qubits)

For each: decompose to the FPGA kernel's supported basis (u, cx), run on the
Alveo U55C via FPGAMPSSimulator, reconstruct the statevector from the FPGA
run, and compare against Qiskit's exact Statevector for the same circuit
(fidelity) plus, where applicable, the physical result the notebook printed
(Hamiltonian eigenvalue / ground-state energy).
"""
from __future__ import annotations

import sys
import time
import numpy as np

sys.path.insert(0, '.')


def fidelity(sv_a: np.ndarray, sv_b: np.ndarray) -> float:
    ov = np.vdot(sv_a, sv_b)
    return float(np.abs(ov) ** 2)


def decompose_to_basis(circuit):
    from qiskit import transpile
    return transpile(circuit, basis_gates=['u', 'cx'], optimization_level=0)


def section(title):
    print('\n' + '=' * 78)
    print(title)
    print('=' * 78)


# ─────────────────────────────────────────────────────────────────────────────
def build_fermionic_gaussian_state():
    from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian
    from qiskit_nature.second_q.circuit.library import FermionicGaussianState
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    hermitian_part = np.array([
        [1.0, 2.0, 0.0, 0.0],
        [2.0, 1.0, 2.0, 0.0],
        [0.0, 2.0, 1.0, 2.0],
        [0.0, 0.0, 2.0, 1.0],
    ])
    antisymmetric_part = np.array([
        [0.0, 3.0, 0.0, 0.0],
        [-3.0, 0.0, 3.0, 0.0],
        [0.0, -3.0, 0.0, 3.0],
        [0.0, 0.0, -3.0, 0.0],
    ])
    constant = 4.0

    hamiltonian = QuadraticHamiltonian(
        hermitian_part=hermitian_part,
        antisymmetric_part=antisymmetric_part,
        constant=constant,
    )
    hamiltonian_ferm = hamiltonian.second_q_op()
    tm, orbital_energies, transformed_constant = hamiltonian.diagonalizing_bogoliubov_transform()

    occupied_orbitals = (0, 2)
    eig = np.sum(orbital_energies[list(occupied_orbitals)]) + transformed_constant

    circuit = FermionicGaussianState(tm, occupied_orbitals=occupied_orbitals)
    hamiltonian_jw = JordanWignerMapper().map(hamiltonian_ferm).to_matrix()

    return circuit, hamiltonian_jw, float(eig)


def build_slater_determinant():
    from qiskit_nature.second_q.hamiltonians import QuadraticHamiltonian
    from qiskit_nature.second_q.circuit.library import SlaterDeterminant
    from qiskit_nature.second_q.mappers import JordanWignerMapper

    hermitian_part = np.array([
        [1.0, 2.0, 0.0, 0.0],
        [2.0, 1.0, 2.0, 0.0],
        [0.0, 2.0, 1.0, 2.0],
        [0.0, 0.0, 2.0, 1.0],
    ])
    constant = 4.0

    hamiltonian = QuadraticHamiltonian(hermitian_part=hermitian_part, constant=constant)
    hamiltonian_ferm = hamiltonian.second_q_op()
    tm, orbital_energies, transformed_constant = hamiltonian.diagonalizing_bogoliubov_transform()

    occupied_orbitals = (0, 2)
    eig = np.sum(orbital_energies[list(occupied_orbitals)]) + transformed_constant

    circuit = SlaterDeterminant(tm[list(occupied_orbitals)])
    hamiltonian_jw = JordanWignerMapper().map(hamiltonian_ferm).to_matrix()

    return circuit, hamiltonian_jw, float(eig)


def build_h2_uccsd_vqe():
    from qiskit_nature.units import DistanceUnit
    from qiskit_nature.second_q.drivers import PySCFDriver
    from qiskit_nature.second_q.mappers import JordanWignerMapper
    from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
    from qiskit_algorithms import VQE
    from qiskit_algorithms.optimizers import SLSQP
    from qiskit.primitives import Estimator
    from qiskit_nature.second_q.algorithms import GroundStateEigensolver

    driver = PySCFDriver(
        atom="H 0 0 0; H 0 0 0.735",
        basis="sto3g",
        charge=0,
        spin=0,
        unit=DistanceUnit.ANGSTROM,
    )
    es_problem = driver.run()
    mapper = JordanWignerMapper()

    ansatz = UCCSD(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
        initial_state=HartreeFock(
            es_problem.num_spatial_orbitals,
            es_problem.num_particles,
            mapper,
        ),
    )
    vqe_solver = VQE(Estimator(), ansatz, SLSQP())
    vqe_solver.initial_point = [0.0] * ansatz.num_parameters

    calc = GroundStateEigensolver(mapper, vqe_solver)
    res = calc.solve(es_problem)

    optimal_point = res.raw_result.optimal_point
    bound_circuit = ansatz.assign_parameters(optimal_point)

    return bound_circuit, float(res.total_energies[0].real), float(res.electronic_energies[0].real)


# ─────────────────────────────────────────────────────────────────────────────
def run_on_fpga(sim, circuit_basis, label):
    from fpga_mps_simulator import _pauli_expectation_sv  # noqa: F401  (kept for parity checks below)

    t0 = time.time()
    result = sim.run(circuit_basis, shots=4096, seed=42)
    dt = time.time() - t0
    sv = result.get_statevector()
    phase = result.get_phase_times()
    print(f"  [{label}] FPGA run wall time: {dt:.3f}s  "
          f"(fpga_kernel={phase.get('fpga_kernel',0):.4f}s, "
          f"hbm_transfer={phase.get('hbm_transfer',0):.4f}s, "
          f"cpu_svd={phase.get('cpu_svd',0):.4f}s)")
    print(f"  [{label}] max bond dim reached: {result.get_max_bond()}")
    return sv


def main():
    section("Connecting to FPGAMPSSimulator (Alveo U55C, engine='fpga')")
    from fpga_mps_simulator import FPGAMPSSimulator
    from qiskit.quantum_info import Statevector

    sim = FPGAMPSSimulator(num_cards=1, chi_max=64, engine='fpga', mode='shots')
    print("Card stats:", sim.get_card_stats())

    results = []

    # ── 1. FermionicGaussianState (11_quadratic_hamiltonian_and_slater_determinants.ipynb) ──
    section("1/3  FermionicGaussianState eigenstate  (notebook 11)")
    circuit, ham_jw, eig = build_fermionic_gaussian_state()
    circuit_basis = decompose_to_basis(circuit)
    print(f"  qubits={circuit.num_qubits}  gates(after decompose)="
          f"{len(circuit_basis.data)}  expected eigenvalue={eig:.6f}")

    sv_exact = np.array(Statevector(circuit_basis))
    sv_fpga = run_on_fpga(sim, circuit_basis, "FermionicGaussianState")

    fid = fidelity(sv_fpga, sv_exact)
    eig_fpga = float(np.real(sv_fpga.conj() @ ham_jw @ sv_fpga))
    eig_exact = float(np.real(sv_exact.conj() @ ham_jw @ sv_exact))
    print(f"  fidelity(FPGA, exact Statevector)      = {fid:.10f}")
    print(f"  <H> from FPGA statevector               = {eig_fpga:.8f}  (expected {eig:.8f})")
    print(f"  <H> from qiskit exact Statevector        = {eig_exact:.8f}")
    results.append(("FermionicGaussianState (nb 11)", fid, eig_fpga, eig))

    # ── 2. SlaterDeterminant (11_quadratic_hamiltonian_and_slater_determinants.ipynb) ──
    section("2/3  SlaterDeterminant eigenstate  (notebook 11)")
    circuit, ham_jw, eig = build_slater_determinant()
    circuit_basis = decompose_to_basis(circuit)
    print(f"  qubits={circuit.num_qubits}  gates(after decompose)="
          f"{len(circuit_basis.data)}  expected eigenvalue={eig:.6f}")

    sv_exact = np.array(Statevector(circuit_basis))
    sv_fpga = run_on_fpga(sim, circuit_basis, "SlaterDeterminant")

    fid = fidelity(sv_fpga, sv_exact)
    eig_fpga = float(np.real(sv_fpga.conj() @ ham_jw @ sv_fpga))
    eig_exact = float(np.real(sv_exact.conj() @ ham_jw @ sv_exact))
    print(f"  fidelity(FPGA, exact Statevector)      = {fid:.10f}")
    print(f"  <H> from FPGA statevector               = {eig_fpga:.8f}  (expected {eig:.8f})")
    print(f"  <H> from qiskit exact Statevector        = {eig_exact:.8f}")
    results.append(("SlaterDeterminant (nb 11)", fid, eig_fpga, eig))

    # ── 3. UCCSD+HF VQE ground state for H2 (03_ground_state_solvers.ipynb) ──
    section("3/3  UCCSD+HartreeFock VQE ground state, H2/STO-3G  (notebook 03)")
    print("  Re-running VQE (SLSQP, JordanWignerMapper) to get optimal parameters...")
    circuit, total_energy_ref, electronic_energy_ref = build_h2_uccsd_vqe()
    circuit_basis = decompose_to_basis(circuit)
    print(f"  qubits={circuit.num_qubits}  gates(after decompose)="
          f"{len(circuit_basis.data)}")
    print(f"  notebook-reported total ground-state energy = {total_energy_ref:.8f} Ha")

    sv_exact = np.array(Statevector(circuit_basis))
    sv_fpga = run_on_fpga(sim, circuit_basis, "UCCSD H2 VQE")

    fid = fidelity(sv_fpga, sv_exact)
    print(f"  fidelity(FPGA, exact Statevector)      = {fid:.10f}")
    results.append(("UCCSD H2 VQE ground state (nb 03)", fid, None, None))

    section("SUMMARY")
    for name, fid, got, expected in results:
        if got is not None:
            print(f"  {name:38s} fidelity={fid:.8f}  <H>_fpga={got:.6f}  expected={expected:.6f}  "
                  f"|Δ|={abs(got-expected):.2e}")
        else:
            print(f"  {name:38s} fidelity={fid:.8f}")

    sim.shutdown()


if __name__ == '__main__':
    main()
