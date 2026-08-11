import sys
import os
import time
from qiskit import QuantumCircuit
from fpga_simulator_v07 import FPGASimulator


current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

xclbin = os.path.join(current_dir, 'quantum_fpga_kernel_complex128.xclbin')

backend = FPGASimulator(xclbin_path=xclbin)


print(f"\nTesting qiskit MCX Gate...")

max_qubits = 10

for n in range(2, max_qubits + 1):

    qc = QuantumCircuit(n, n)
    qc.x(range(n-1))
    qc.mcx(list(range(n-1)), n-1)
    qc.measure(range(n), range(n))

    start_time = time.time()
    result = backend.run(qc, shots=1000)
    exec_time = time.time() - start_time
    counts = result.get_counts()
    print(f"  Sample results: {list(counts.items())[:8]}")
    print(f"  Execution Time: {exec_time} sec")