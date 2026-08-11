import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Quantum Circuit
num_qubits = 3
qc = QuantumCircuit(num_qubits, num_qubits)
qc.h(0)


qc.measure(range(num_qubits), range(num_qubits))


# Measure both qubits
qc.measure(range(num_qubits), range(num_qubits))

# Run on AerSimulator
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
result = simulator.run(compiled_circuit).result()

# Get counts
counts = result.get_counts()
print("Measurement results:", counts)
