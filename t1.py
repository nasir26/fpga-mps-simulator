import math
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

print(f"\nTesting quantum circuit...")

# Quantum Circuit
qc = QuantumCircuit(1, 2)
qc.h(0)
qc.measure(0, 0)
qc.h(0)
qc.measure(0, 1)


simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)

job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
print(result)

counts = result.get_counts(compiled_circuit)
print("Counts:", counts)