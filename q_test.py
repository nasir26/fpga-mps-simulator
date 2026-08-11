# from qiskit import QuantumCircuit
# # To run:
# from qiskit_aer import Aer
# from qiskit import transpile

# qc = QuantumCircuit(1, 3)
# qc.h(0)
# qc.measure(0, 0)
# qc.h(0)
# qc.measure(0, 1)
# qc.h(0)
# qc.measure(0, 2)

# print(qc)

# job = Aer.get_backend("aer_simulator").run(transpile(qc, Aer.get_backend("aer_simulator")), shots=1)
# result = job.result()
# print(result.get_counts())



from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit import transpile
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

qc = QuantumCircuit(1, 3)
qc.h(0)
qc.measure(0, 0)
qc.h(0)
qc.measure(0, 1)
qc.h(0)
qc.measure(0, 2)

print(qc)

backend = Aer.get_backend("aer_simulator")
job = backend.run(transpile(qc, backend), shots=10)
result = job.result()
counts = result.get_counts()
print(counts)

plot_histogram(counts)
plt.show()