"""Isolate why fidelity was low for the Nature-notebook circuits: test the
FPGA pipeline on small, hand-verifiable circuits of increasing complexity."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from fpga_mps_simulator import FPGAMPSSimulator


def fidelity(a, b):
    return float(np.abs(np.vdot(a, b)) ** 2)


def test(sim, name, qc):
    sv_exact = np.array(Statevector(qc))
    res = sim.run(qc, shots=1024, seed=1)
    sv_fpga = res.get_statevector()
    fid = fidelity(sv_fpga, sv_exact)
    print(f"{name:40s} n={qc.num_qubits} gates={len(qc.data)} fidelity={fid:.8f}")
    if fid < 0.999:
        print("  exact:", np.round(sv_exact, 4))
        print("  fpga :", np.round(sv_fpga, 4))
    return fid


def main():
    sim = FPGAMPSSimulator(num_cards=1, chi_max=64, engine='fpga', mode='shots')

    qc = QuantumCircuit(2)
    qc.h(0); qc.cx(0, 1)
    test(sim, "Bell state (adjacent cx)", qc)

    qc = QuantumCircuit(4)
    qc.h(0)
    for i in range(3):
        qc.cx(i, i + 1)
    test(sim, "GHZ-4 (adjacent cx chain)", qc)

    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 3)
    test(sim, "cx(0,3) non-adjacent (routing)", qc)

    qc = QuantumCircuit(1)
    qc.u(0.7, 1.3, -0.4, 0)
    test(sim, "single u(theta,phi,lambda)", qc)

    qc = QuantumCircuit(2)
    qc.u(0.7, 1.3, -0.4, 0)
    qc.u(-0.9, 0.2, 2.1, 1)
    qc.cx(0, 1)
    qc.u(0.3, -0.6, 1.1, 0)
    test(sim, "u + cx + u (adjacent)", qc)

    qc = QuantumCircuit(4)
    qc.u(0.7, 1.3, -0.4, 0)
    qc.u(-0.9, 0.2, 2.1, 2)
    qc.cx(0, 2)
    test(sim, "u + cx(0,2) non-adjacent", qc)

    sim.shutdown()


if __name__ == '__main__':
    main()
