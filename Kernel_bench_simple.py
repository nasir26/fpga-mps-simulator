#!/usr/bin/env python3
"""
kernel_bench_simple.py
=======================
Minimal FPGA vs CPU vs GPU benchmark, KERNEL EXECUTION TIME ONLY.

  * No plots, no CSV files — just a plain-text table printed to the
    terminal, once per algorithm.
  * Qiskit's circuit build + transpile cost is paid ONCE per (hardware,
    qubit count), *outside* the timed loop, for both CPU and GPU — so
    the numbers below are pure `backend.run()` execution time
    (`result.results[0].time_taken`, the same figure Qiskit itself
    reports), not "time to build a circuit object in Python".
  * FPGA uses its own on-device kernel timer (`res['timing']['kernel_us']`)
    for the same reason — it's the FPGA's report of how long the kernel
    itself ran, not how long the Python call around it took.
  * Every number is the MEAN of several timed repeats of the *same*
    already-built circuit (a couple of untimed warm-up runs happen
    first), with the population std-dev printed alongside it, so a
    single lucky/unlucky run can't skew the result.

Algorithms benchmarked:
  QFT           1..MAX_QFT_QUBITS qubits
  Grover        2..MAX_GROVER_QUBITS qubits   (marks |11...1>)
  QFT + IQFT    1..MAX_QFT_IQFT_QUBITS qubits (round trip back to |0...0>)

Run from the same directory as `fpga_simulator_v07.py` and
`quantum_fpga_kernel_complex128.xclbin` (same layout as the other test
scripts). If the FPGA module or a GPU device isn't available, that
column is skipped automatically and printed as "n/a" — the script
never hard-fails just because one hardware is missing.
"""

import math
import statistics
import time

# ─────────────────────────────────────────────────────────────────────────
#  Knobs — change these, nothing else, to tune runtime vs. confidence
# ─────────────────────────────────────────────────────────────────────────
REPEATS_DEFAULT = 7     # timed repeats for "cheap" (low-qubit) circuits
REPEATS_MEDIUM  = 5     # timed repeats once circuits get more expensive
REPEATS_LARGE   = 3     # timed repeats for the most expensive circuits
WARMUP          = 2     # untimed warm-up runs before every timed batch

MAX_QFT_QUBITS       = 20
MAX_GROVER_QUBITS    = 15
MAX_QFT_IQFT_QUBITS  = 10

XCLBIN = "quantum_fpga_kernel_complex128.xclbin"


def repeats_for(nq):
    """Fewer repeats for expensive (large-qubit) circuits, so a full sweep
    up to 20 qubits doesn't take forever — still always > 1, so it's
    always an average, never a single sample."""
    if nq >= 14:
        return REPEATS_LARGE
    if nq >= 10:
        return REPEATS_MEDIUM
    return REPEATS_DEFAULT


# ─────────────────────────────────────────────────────────────────────────
#  Gate builders — byte-for-byte identical circuit across all 3 hardwares
# ─────────────────────────────────────────────────────────────────────────
def qft_gates(n, inverse=False):
    gates = []
    if not inverse:
        for i in range(n):
            gates.append(('h', [i], None))
            for j in range(i + 1, n):
                angle = math.pi / (2 ** (j - i))
                gates.append(('cp', [j, i], [angle]))
        for i in range(n // 2):
            gates.append(('swap', [i, n - 1 - i], None))
    else:
        for i in range(n // 2):
            gates.append(('swap', [i, n - 1 - i], None))
        for i in reversed(range(n)):
            for j in reversed(range(i + 1, n)):
                angle = -math.pi / (2 ** (j - i))
                gates.append(('cp', [j, i], [angle]))
            gates.append(('h', [i], None))
    return gates


def qft_iqft_gates(n):
    """QFT immediately followed by its inverse -> should land back on |0..0>."""
    return qft_gates(n, inverse=False) + qft_gates(n, inverse=True)


GROVER_BASIS_GATES = ['h', 'x', 'cx', 'cz', 'cp', 'swap', 'rz', 'ry', 'rx',
                       's', 'sdg', 't', 'tdg', 'p']


def _apply_mcz(qc, n):
    """Phase-flip |11...1>. n=1 -> Z, n=2 -> CZ, else H+MCX+H (no ancilla)."""
    if n == 1:
        qc.z(0)
    elif n == 2:
        qc.cz(0, 1)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)


def grover_gates(n):
    """Grover's search marking |11...1>, optimal iteration count, transpiled
    once to GROVER_BASIS_GATES and replayed as a plain gate list so every
    hardware executes the identical gate sequence."""
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(n)
    qc.h(range(n))
    theta = math.asin(1.0 / math.sqrt(2 ** n))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    for _ in range(iterations):
        _apply_mcz(qc, n)
        qc.h(range(n))
        qc.x(range(n))
        _apply_mcz(qc, n)
        qc.x(range(n))
        qc.h(range(n))

    tqc = transpile(qc, basis_gates=GROVER_BASIS_GATES, optimization_level=1)
    qidx = {q: i for i, q in enumerate(tqc.qubits)}
    gates = []
    for ci in tqc.data:
        instr, qargs = ci.operation, ci.qubits
        gates.append((instr.name, [qidx[q] for q in qargs],
                      list(instr.params) if instr.params else None))
    return gates


def gates_to_circuit(nq, gates):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(nq)
    builders = {
        'h':    lambda q, p: qc.h(q[0]),
        'x':    lambda q, p: qc.x(q[0]),
        'y':    lambda q, p: qc.y(q[0]),
        'z':    lambda q, p: qc.z(q[0]),
        's':    lambda q, p: qc.s(q[0]),
        'sdg':  lambda q, p: qc.sdg(q[0]),
        't':    lambda q, p: qc.t(q[0]),
        'tdg':  lambda q, p: qc.tdg(q[0]),
        'id':   lambda q, p: qc.id(q[0]),
        'rx':   lambda q, p: qc.rx(p[0], q[0]),
        'ry':   lambda q, p: qc.ry(p[0], q[0]),
        'rz':   lambda q, p: qc.rz(p[0], q[0]),
        'p':    lambda q, p: qc.p(p[0], q[0]),
        'cx':   lambda q, p: qc.cx(q[0], q[1]),
        'cz':   lambda q, p: qc.cz(q[0], q[1]),
        'cp':   lambda q, p: qc.cp(p[0], q[0], q[1]),
        'swap': lambda q, p: qc.swap(q[0], q[1]),
        'ccx':  lambda q, p: qc.ccx(q[0], q[1], q[2]),
    }
    for name, qubits, params in gates:
        builder = builders.get(name)
        if builder is None:
            raise ValueError(f"gates_to_circuit: unsupported gate '{name}'")
        builder(qubits, params or [])
    return qc


# ─────────────────────────────────────────────────────────────────────────
#  Backend bring-up
# ─────────────────────────────────────────────────────────────────────────
def init_fpga():
    try:
        from fpga_simulator_v07 import QuantumFPGACircuit
    except ImportError as e:
        print(f"  [skip] FPGA: cannot import fpga_simulator_v07 ({e})")
        return None
    try:
        fpga = QuantumFPGACircuit(XCLBIN)
        print("  [ok]   FPGA ready")
        return fpga
    except Exception as e:
        print(f"  [skip] FPGA: init failed ({e})")
        return None


def init_aer(device):
    try:
        from qiskit_aer import AerSimulator
        from qiskit import QuantumCircuit, transpile
    except ImportError as e:
        print(f"  [skip] {device}: cannot import qiskit_aer ({e})")
        return None
    try:
        backend = AerSimulator(method='statevector', device=device)
        probe = QuantumCircuit(2)
        probe.h(0)
        probe.save_statevector()
        backend.run(transpile(probe, backend, optimization_level=0), shots=1).result()
        print(f"  [ok]   {device} ready")
        return backend
    except Exception as e:
        print(f"  [skip] {device}: unavailable ({e})")
        return None


# ─────────────────────────────────────────────────────────────────────────
#  Kernel-only timing helpers
#  (build/transpile happens ONCE, before the timed loop, for CPU & GPU;
#   FPGA reports its own kernel time, so no build/transpile step exists)
# ─────────────────────────────────────────────────────────────────────────
def time_fpga(fpga, nq, gates):
    n = repeats_for(nq)
    for _ in range(WARMUP):
        fpga.run_circuit(nq, gates, shots=0, validate=False, circuit_name="warmup")
    samples = []
    for _ in range(n):
        res = fpga.run_circuit(nq, gates, shots=0, validate=False, circuit_name="bench")
        samples.append(res['timing']['kernel_us'] / 1e3)   # us -> ms
    return samples


def time_aer(backend, nq, gates):
    from qiskit import transpile
    n = repeats_for(nq)

    qc = gates_to_circuit(nq, gates)
    qc.save_statevector()
    tqc = transpile(qc, backend, optimization_level=0)     # <-- paid once, NOT timed

    for _ in range(WARMUP):
        backend.run(tqc, shots=1).result()

    samples = []
    for _ in range(n):
        res = backend.run(tqc, shots=1).result()
        samples.append(res.results[0].time_taken * 1e3)    # Qiskit-reported kernel time -> ms
    return samples


def mean_std(samples):
    return statistics.mean(samples), (statistics.pstdev(samples) if len(samples) > 1 else 0.0)


# ─────────────────────────────────────────────────────────────────────────
#  Sweep + plain-text report (no plots, no CSV)
# ─────────────────────────────────────────────────────────────────────────
def cell(row, key):
    if key not in row:
        return "n/a"
    m, s = row[key]
    return f"{m:9.4f} ±{s:6.3f}"


def run_sweep(title, min_q, max_q, gate_fn, fpga, cpu, gpu):
    print(f"\n{'=' * 82}")
    print(f"  {title}")
    print(f"  (kernel time only, mean ± population-stdev of N timed repeats, N per repeats_for())")
    print(f"{'=' * 82}")
    print(f"{'Qubits':>6} | {'FPGA (ms)':>17} | {'CPU (ms)':>17} | {'GPU (ms)':>17} | {'N':>3} | Fastest")
    print('-' * 82)

    for nq in range(min_q, max_q + 1):
        gates = gate_fn(nq)
        row = {}
        if fpga is not None:
            row['FPGA'] = mean_std(time_fpga(fpga, nq, gates))
        if cpu is not None:
            row['CPU'] = mean_std(time_aer(cpu, nq, gates))
        if gpu is not None:
            row['GPU'] = mean_std(time_aer(gpu, nq, gates))

        winner = min(row, key=lambda k: row[k][0]) if row else '-'
        print(f"{nq:>6} | {cell(row,'FPGA'):>17} | {cell(row,'CPU'):>17} | "
              f"{cell(row,'GPU'):>17} | {repeats_for(nq):>3} | {winner}")


def main():
    t_start = time.perf_counter()
    print("=" * 82)
    print("  Simple kernel-only benchmark : FPGA vs CPU vs GPU")
    print(f"  Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 82)
    print("\nInitialising hardware ...")
    fpga = init_fpga()
    cpu = init_aer('CPU')
    gpu = init_aer('GPU')

    run_sweep(f"QFT scaling (1-{MAX_QFT_QUBITS} qubits)",
              1, MAX_QFT_QUBITS, qft_gates, fpga, cpu, gpu)

    run_sweep(f"Grover's algorithm scaling (2-{MAX_GROVER_QUBITS} qubits)",
              2, MAX_GROVER_QUBITS, grover_gates, fpga, cpu, gpu)

    run_sweep(f"QFT + IQFT round trip (1-{MAX_QFT_IQFT_QUBITS} qubits)",
              1, MAX_QFT_IQFT_QUBITS, qft_iqft_gates, fpga, cpu, gpu)

    print(f"\nTotal wall time: {time.perf_counter() - t_start:.1f} s")
    print("Done.")


if __name__ == '__main__':
    main()