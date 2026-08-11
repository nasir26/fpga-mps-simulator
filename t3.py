# Quantum Circuit
import sys
import os
import time
import json
import numpy as np
from qiskit import QuantumCircuit

from fpga_simulator_v07 import FPGASimulator

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

xclbin = os.path.join(current_dir, 'quantum_fpga_kernel_complex128.xclbin')

backend = FPGASimulator(xclbin_path=xclbin)

# Quantum Circuit
qc = QuantumCircuit(1, 2)
qc.h(0)
qc.measure(0, 0)
qc.h(0)
qc.measure(0, 1)

start_time = time.time()
result = backend.run(qc, shots=1024)
exec_time = time.time() - start_time

counts = result.get_counts()
# sv = get_statevector(result)

print(f"  Counts: {list(counts.items())[:8]}")
print(f"  Execution Time: {exec_time} sec")
# print(f"  Statevector: {sv}")