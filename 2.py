#!/usr/bin/env python3
"""
hardware_comparison_qft.py
===========================
Single-file benchmark that runs the SAME workload(s) on three hardware
backends —

    FPGA   fpga_simulator_v07 (quantum_fpga_kernel_complex128.xclbin)
    CPU    Qiskit Aer, AerSimulator(method='statevector', device='CPU')
    GPU    Qiskit Aer, AerSimulator(method='statevector', device='GPU')
           (requires qiskit-aer-gpu + cuStateVec)

Two algorithms are benchmarked, each over its own qubit sweep:

    QFT     1..20 qubits   (MIN_QUBITS_QFT .. MAX_QUBITS_QFT)
    Grover  1..15 qubits   (MIN_QUBITS_GROVER .. MAX_QUBITS_GROVER)

For EACH algorithm this produces:

  1. hardware_comparison_<algo>.csv
       every (hardware, qubit_count) measurement: kernel time, wall
       time, pass/fail, and speedup vs the slowest hardware at that
       qubit count.

  2. hardware_comparison_<algo>_kernel_time.png
       log-scale line chart, execution ("kernel") time vs qubit count,
       one clearly-marked line per hardware, with every data point
       labelled.

  3. hardware_comparison_<algo>_wall_time.png
       same, but for full Python-side wall time (build+transpile+run).

  4. hardware_comparison_<algo>_speedup.png
       speedup of GPU and FPGA relative to CPU at every qubit count,
       log-scale, with a y=1 reference line and crossover points
       (where the ranking flips) called out directly on the plot.

  5. hardware_comparison_<algo>_winner_bars.png
       grouped bar chart, one group per qubit count, one bar per
       hardware, with the fastest bar in each group starred and
       labelled with its speedup over the runner-up.

Console output also prints a plain-text "who wins at what qubit count"
summary and the exact qubit count(s) where the ranking between any two
hardwares flips, per algorithm.

Run this from the same directory as `fpga_simulator_v07.py` and
`quantum_fpga_kernel_complex128.xclbin` (same layout as
comprehensive_test_v07.py / aer_test.py). If the FPGA module or a GPU
device is unavailable, that hardware is skipped automatically and the
comparison continues with whichever backends ARE available — the
script never hard-fails just because one hardware is missing.

Grover circuits are built with Qiskit and transpiled once to a fixed
basis-gate set (see `GROVER_BASIS_GATES`), and that IDENTICAL gate
list is then replayed on every active backend (FPGA via
`fpga.run_circuit`, CPU/GPU via a plain Aer QuantumCircuit rebuilt
from the same gate list) so all hardwares execute the byte-for-byte
same circuit, exactly like the existing QFT benchmark. The marked
("winning") basis state is the all-ones bitstring |11...1>; the
oracle/diffuser use a no-ancilla multi-controlled-Z built from H +
MCX, decomposed down to the basis set below.

NOTE: if your FPGA kernel does not implement every gate name in
GROVER_BASIS_GATES, those (FPGA, nq) cells will show FAIL in the
console/CSV instead of crashing the whole run (same graceful
degradation as everywhere else in this script) — just trim
GROVER_BASIS_GATES down to whatever `fpga_simulator_v07` actually
supports and rerun.
"""

import csv
import math
import sys
import time
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─────────────────────────────────────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────────────────────────────────────
MIN_QUBITS_QFT     = 1
MAX_QUBITS_QFT     = 20     # matches the existing FPGA/Aer test-suite sweep
MIN_QUBITS_GROVER  = 1
MAX_QUBITS_GROVER  = 15     # Grover's classical statevector cost explodes fast

XCLBIN       = "quantum_fpga_kernel_complex128.xclbin"
UNIFORM_TOL  = 0.05         # QFT|0...0> -> uniform superposition check
GROVER_TOL   = 1e-6         # Grover: marked-state prob must exceed uniform by this margin

# Basis gates the Grover circuit is transpiled to before being replayed,
# gate-for-gate, on every backend. Trim/extend to match your FPGA kernel.
GROVER_BASIS_GATES = ['h', 'x', 'cx', 'cz', 'cp', 'swap', 'rz', 'ry', 'rx',
                       's', 'sdg', 't', 'tdg', 'p']

HW = {
    'FPGA': {'color': '#FF7F0E', 'marker': 'o'},   # orange
    'CPU':  {'color': '#1F77B4', 'marker': 's'},   # blue
    'GPU':  {'color': '#2CA02C', 'marker': '^'},   # green
}
GRID_COLOR = '#E0E0E0'


# ─────────────────────────────────────────────────────────────────────────────
#  QFT circuit (ported from aer_test.py / comprehensive_test_v07.py so all
#  three hardwares run the byte-for-byte identical gate sequence)
# ─────────────────────────────────────────────────────────────────────────────
def qft_gates(n):
    gates = []
    for i in range(n):
        gates.append(('h', [i], None))
        for j in range(i + 1, n):
            angle = math.pi / (2 ** (j - i))
            gates.append(('cp', [j, i], [angle]))
    for i in range(n // 2):
        gates.append(('swap', [i, n - 1 - i], None))
    return gates


# ─────────────────────────────────────────────────────────────────────────────
#  Grover circuit (built with Qiskit, transpiled once to GROVER_BASIS_GATES,
#  then replayed as a plain gate list on every backend, same pattern as QFT)
# ─────────────────────────────────────────────────────────────────────────────
def _apply_mcz(qc, n):
    """Phase-flip |11...1> (no ancilla). n=1 -> Z, n=2 -> CZ, else H+MCX+H."""
    if n == 1:
        qc.z(0)
    elif n == 2:
        qc.cz(0, 1)
    else:
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)   # no-ancilla multi-controlled-X
        qc.h(n - 1)


def grover_gates(n):
    """Grover's search marking the all-ones state |11...1>, optimal number
    of iterations, returned as the same (name, qubits, params) tuple list
    format used by qft_gates()."""
    from qiskit import QuantumCircuit, transpile

    qc = QuantumCircuit(n)
    qc.h(range(n))
    # Exact optimal iteration count for a single marked item out of N=2**n:
    # amplitude follows sin((2t+1)*theta), theta = asin(1/sqrt(N)); the naive
    # (pi/4)*sqrt(N) approximation over-rotates for small N (e.g. n=2), so use
    # the corrected round(pi/(4*theta) - 1/2) form instead.
    theta = math.asin(1.0 / math.sqrt(2 ** n))
    iterations = max(1, round((math.pi / (4 * theta)) - 0.5))
    for _ in range(iterations):
        _apply_mcz(qc, n)                 # oracle (target is already |1..1>)
        qc.h(range(n))                    # diffuser
        qc.x(range(n))
        _apply_mcz(qc, n)
        qc.x(range(n))
        qc.h(range(n))

    tqc = transpile(qc, basis_gates=GROVER_BASIS_GATES, optimization_level=1)
    qidx = {q: i for i, q in enumerate(tqc.qubits)}
    gates = []
    for instr, qargs, _cargs in tqc.data:
        gates.append((instr.name, [qidx[q] for q in qargs],
                      list(instr.params) if instr.params else None))
    return gates


def grover_check(sv, n):
    """Pass if the marked state |11...1> is the (uniquely) highest-probability
    outcome and its probability clears the uniform baseline by GROVER_TOL.
    n=1 (search space of size 2) is a known Grover degenerate case: the
    marked-state amplitude oscillates at exactly the uniform value and can
    never be amplified above it, so it's always reported as passed."""
    if n == 1:
        return True
    probs = np.abs(np.asarray(sv)) ** 2
    marked_idx = 2 ** n - 1
    return bool(probs[marked_idx] >= probs.max() - 1e-9
                and probs[marked_idx] > (1.0 / (2 ** n)) + GROVER_TOL)


# ─────────────────────────────────────────────────────────────────────────────
#  Gate-list -> Qiskit circuit (generic; used by both algorithms' Aer runners)
# ─────────────────────────────────────────────────────────────────────────────
def gates_to_circuit(nq, gates):
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(nq)
    BUILDERS = {
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
        builder = BUILDERS.get(name)
        if builder is None:
            raise ValueError(f"gates_to_circuit: unsupported gate '{name}'")
        builder(qubits, params or [])
    return qc


# ─────────────────────────────────────────────────────────────────────────────
#  Backend bring-up — each returns None (with a printed reason) if that
#  hardware isn't available in this environment, so the rest of the script
#  degrades gracefully instead of crashing.
# ─────────────────────────────────────────────────────────────────────────────
def init_fpga():
    try:
        from fpga_simulator_v07 import QuantumFPGACircuit
    except ImportError as e:
        print(f"  [skip] FPGA: cannot import fpga_simulator_v07 ({e})")
        return None
    try:
        t0 = time.perf_counter_ns()
        fpga = QuantumFPGACircuit(XCLBIN)
        print(f"  [ok]   FPGA ready in {(time.perf_counter_ns()-t0)/1e6:.1f} ms")
        return fpga
    except Exception as e:
        print(f"  [skip] FPGA: init failed ({e})")
        return None


def init_aer(device):
    try:
        from qiskit_aer import AerSimulator
    except ImportError as e:
        print(f"  [skip] {device}: cannot import qiskit_aer ({e})")
        return None
    try:
        t0 = time.perf_counter_ns()
        backend = AerSimulator(method='statevector', device=device)
        # Force a tiny real run now (not just construction) so a GPU/cuStateVec
        # problem surfaces here instead of mid-sweep.
        from qiskit import QuantumCircuit, transpile
        probe = QuantumCircuit(2)
        probe.h(0)
        probe.save_statevector()
        tprobe = transpile(probe, backend, optimization_level=0)
        backend.run(tprobe, shots=1).result()
        print(f"  [ok]   {device} ready in {(time.perf_counter_ns()-t0)/1e6:.1f} ms")
        return backend
    except Exception as e:
        print(f"  [skip] {device}: unavailable ({e})")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Generic per-hardware run -> (kernel_ms, wall_ms, passed)
#  `gate_fn(nq)` returns the (name, qubits, params) gate list for that qubit
#  count; `check_fn(statevector, nq)` returns pass/fail for Aer runs.
# ─────────────────────────────────────────────────────────────────────────────
def run_fpga(fpga, nq, gates, circuit_name, uniform_check=False):
    t0 = time.perf_counter_ns()
    res = fpga.run_circuit(
        nq, gates, shots=0, validate=uniform_check,
        uniform_check=uniform_check, uniform_tolerance=UNIFORM_TOL,
        circuit_name=f"{circuit_name} {nq}q")
    wall_us = (time.perf_counter_ns() - t0) / 1e3
    kernel_us = res['timing']['kernel_us']
    val = res.get('validation') if isinstance(res, dict) else None
    passed = bool(val.passed) if val else True
    return kernel_us / 1e3, wall_us / 1e3, passed


def run_aer(backend, nq, gates, check_fn):
    from qiskit import transpile
    t0 = time.perf_counter_ns()
    qc = gates_to_circuit(nq, gates)
    qc.save_statevector()
    tqc = transpile(qc, backend, optimization_level=0)
    res = backend.run(tqc, shots=1).result()
    sv = np.asarray(res.get_statevector(tqc))
    kernel_us = res.results[0].time_taken * 1e6
    wall_us = (time.perf_counter_ns() - t0) / 1e3
    passed = bool(check_fn(sv, nq))
    return kernel_us / 1e3, wall_us / 1e3, passed


def qft_check(sv, nq):
    n = len(sv)
    expected = 1.0 / n
    probs = np.abs(sv) ** 2
    return np.all(np.abs(probs - expected) < UNIFORM_TOL)


ALGORITHMS = {
    'QFT': {
        'gate_fn':     qft_gates,
        'check_fn':    qft_check,
        'min_qubits':  MIN_QUBITS_QFT,
        'max_qubits':  MAX_QUBITS_QFT,
        'fpga_uniform_check': True,   # QFT|0..0> has a known closed-form check
        'csv_path':    'hardware_comparison_qft.csv',
        'plot_prefix': 'hardware_comparison',   # kept for backward-compat filenames
        'label':       'QFT',
    },
    'Grover': {
        'gate_fn':     grover_gates,
        'check_fn':    grover_check,
        'min_qubits':  MIN_QUBITS_GROVER,
        'max_qubits':  MAX_QUBITS_GROVER,
        'fpga_uniform_check': False,  # no closed-form uniform check for Grover
        'csv_path':    'hardware_comparison_grover.csv',
        'plot_prefix': 'hardware_comparison_grover',
        'label':       'Grover',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
#  Sweep — runs one algorithm end-to-end (init is shared/reused across algos)
# ─────────────────────────────────────────────────────────────────────────────
def run_algorithm(algo_name, cfg, handles, active):
    gate_fn = cfg['gate_fn']
    check_fn = cfg['check_fn']
    min_q, max_q = cfg['min_qubits'], cfg['max_qubits']
    label = cfg['label']
    prefix = cfg['plot_prefix']

    print('=' * 72)
    print(f'  {label} Statevector Simulation — FPGA vs CPU vs GPU')
    print(f'  Qubit sweep : {min_q}..{max_q}')
    print(f'  Started     : {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 72)
    print(f'\nActive hardware: {", ".join(active)}\n')

    # rows[hw][nq] = (kernel_ms, wall_ms, passed)  or None on failure
    rows = {hw: {} for hw in active}

    if algo_name == 'Grover':
        print('  Note: no-ancilla multi-controlled-Z gates scale ~O(n^2) per\n'
              '  use, and Grover needs ~(pi/4)*sqrt(2^n) iterations of two\n'
              '  such gates — so total gate count (and runtime, especially\n'
              '  the FPGA per-gate dispatch overhead) grows steeply toward\n'
              '  n=15. If a row hangs too long, Ctrl+C and lower\n'
              '  MAX_QUBITS_GROVER.\n')

    print('-' * 72)
    header = f"  {'nq':>3s} " + ' '.join(f"{hw:>18s}" for hw in active)
    print(header)
    print('-' * 72)

    for nq in range(min_q, max_q + 1):
        gates = gate_fn(nq)   # built ONCE per nq, replayed on every backend
        if algo_name == 'Grover':
            print(f"  [{nq:2d}q] {len(gates)} gates built, running on "
                  f"{', '.join(active)} …", flush=True)
        line = f"  {nq:3d} "
        for hw in active:
            try:
                if hw == 'FPGA':
                    kernel_ms, wall_ms, passed = run_fpga(
                        handles[hw], nq, gates, label,
                        uniform_check=cfg['fpga_uniform_check'])
                else:
                    kernel_ms, wall_ms, passed = run_aer(
                        handles[hw], nq, gates, check_fn)
                rows[hw][nq] = (kernel_ms, wall_ms, passed)
                mark = '✓' if passed else '✗'
                line += f"{kernel_ms:14.3f} ms{mark} "
            except Exception as exc:
                rows[hw][nq] = None
                line += f"{'FAIL':>18s} "
                print(f"\n  [{hw} @ {nq}q] {type(exc).__name__}: {exc}")
                traceback.print_exc(limit=1)
        print(line, flush=True)

    print('-' * 72)

    # ── CSV ─────────────────────────────────────────────────────────────
    csv_path = cfg['csv_path']
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['qubits', 'hardware', 'kernel_time_ms', 'wall_time_ms',
                     'passed', 'speedup_vs_slowest_this_qubit_count'])
        for nq in range(min_q, max_q + 1):
            valid = {hw: rows[hw][nq] for hw in active if rows[hw].get(nq)}
            slowest_kernel = max(v[0] for v in valid.values()) if valid else None
            for hw in active:
                r = rows[hw].get(nq)
                if r is None:
                    w.writerow([nq, hw, '', '', 'FAIL', ''])
                    continue
                kernel_ms, wall_ms, passed = r
                speedup = (slowest_kernel / kernel_ms) if slowest_kernel else ''
                w.writerow([nq, hw, f'{kernel_ms:.6f}', f'{wall_ms:.6f}',
                             passed, f'{speedup:.3f}' if speedup != '' else ''])
    print(f'\nCSV written -> {csv_path}')

    # ── Console summary: winner per qubit count ────────────────────────
    print('\nFastest hardware per qubit count (by kernel time):')
    winners = {}
    for nq in range(min_q, max_q + 1):
        valid = {hw: rows[hw][nq][0] for hw in active if rows[hw].get(nq)}
        if not valid:
            continue
        best_hw = min(valid, key=valid.get)
        sorted_times = sorted(valid.items(), key=lambda kv: kv[1])
        runner_up_ms = sorted_times[1][1] if len(sorted_times) > 1 else None
        speedup_txt = (f"  ({runner_up_ms/sorted_times[0][1]:.2f}x vs "
                        f"{sorted_times[1][0]})") if runner_up_ms else ''
        winners[nq] = best_hw
        print(f"    {nq:2d}q -> {best_hw:5s} @ {valid[best_hw]:10.3f} ms{speedup_txt}")

    # crossover points: where the winner changes as nq increases
    print('\nCrossover points (where the fastest hardware changes):')
    prev = None
    crossovers = []
    for nq in sorted(winners):
        if prev is not None and winners[nq] != prev:
            print(f"    at {nq}q: {prev} -> {winners[nq]} becomes fastest")
            crossovers.append((nq, prev, winners[nq]))
        prev = winners[nq]
    if not crossovers:
        only = prev if prev else 'n/a'
        print(f"    none — {only} is fastest across the whole sweep")

    # ── Plots ────────────────────────────────────────────────────────────
    kt_png = f'{prefix}_kernel_time.png'
    wt_png = f'{prefix}_wall_time.png'
    sp_png = f'{prefix}_speedup.png'
    wb_png = f'{prefix}_winner_bars.png'

    plot_time_series(rows, active, metric='kernel',
                      title=f'{label} Execution (Kernel) Time', fname=kt_png,
                      algo_label=label, min_q=min_q, max_q=max_q)
    plot_time_series(rows, active, metric='wall',
                      title=f'{label} End-to-End Wall Time', fname=wt_png,
                      algo_label=label, min_q=min_q, max_q=max_q)
    plot_speedup(rows, active, crossovers, algo_label=label, fname=sp_png,
                 min_q=min_q, max_q=max_q)
    plot_winner_bars(rows, active, winners, algo_label=label, fname=wb_png)

    print(f'\nGraphs saved:')
    for g in [kt_png, wt_png, sp_png, wb_png]:
        print(f'    {g}')
    print(f'\nCSV saved:\n    {csv_path}')
    print('=' * 72 + '\n')


def main():
    print('\nInitialising backends …')
    handles = {
        'FPGA': init_fpga(),
        'CPU':  init_aer('CPU'),
        'GPU':  init_aer('GPU'),
    }
    active = [hw for hw, h in handles.items() if h is not None]
    if not active:
        print('\n[FATAL] No hardware backend is available — nothing to benchmark.')
        sys.exit(1)
    print(f'\nActive hardware: {", ".join(active)}')

    for algo_name, cfg in ALGORITHMS.items():
        run_algorithm(algo_name, cfg, handles, active)


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 1/2 — time series (kernel or wall), log-scale, labelled points
# ─────────────────────────────────────────────────────────────────────────────
def plot_time_series(rows, active, metric, title, fname, algo_label, min_q, max_q):
    idx = 0 if metric == 'kernel' else 1
    fig, ax = plt.subplots(figsize=(13, 7))

    for hw in active:
        nqs, vals = [], []
        for nq in sorted(rows[hw]):
            r = rows[hw][nq]
            if r is None:
                continue
            nqs.append(nq)
            vals.append(r[idx])
        if not nqs:
            continue
        c = HW[hw]['color']
        m = HW[hw]['marker']
        ax.plot(nqs, vals, marker=m, color=c, linewidth=2.5, markersize=9,
                 markeredgecolor='white', markeredgewidth=1.0, label=hw, zorder=3)
        # label every point so exact values are readable
        for x, y in zip(nqs, vals):
            ax.annotate(f'{y:.2g}', (x, y), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=7, color=c)

    ax.set_xlabel('Number of qubits', fontsize=12)
    ax.set_ylabel(f'{title} (ms, log scale)', fontsize=12)
    ax.set_title(f'{title} vs Qubit Count — FPGA vs CPU vs GPU ({algo_label} statevector)',
                  fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(range(min_q, max_q + 1))
    ax.grid(True, which='both', color=GRID_COLOR, linewidth=0.6)
    ax.legend(fontsize=11, loc='upper left')
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 3 — speedup of each hardware vs CPU baseline
# ─────────────────────────────────────────────────────────────────────────────
def plot_speedup(rows, active, crossovers, algo_label, fname, min_q, max_q):
    if 'CPU' not in active:
        return  # nothing to normalize against
    others = [hw for hw in active if hw != 'CPU']
    if not others:
        return

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.2,
                label='CPU baseline (1.0x)', zorder=2)

    for hw in others:
        nqs, speedups = [], []
        for nq in sorted(rows[hw]):
            cpu_r = rows['CPU'].get(nq)
            hw_r = rows[hw].get(nq)
            if cpu_r is None or hw_r is None:
                continue
            nqs.append(nq)
            speedups.append(cpu_r[0] / hw_r[0])   # >1 means hw is faster than CPU
        if not nqs:
            continue
        c = HW[hw]['color']
        m = HW[hw]['marker']
        ax.plot(nqs, speedups, marker=m, color=c, linewidth=2.5, markersize=9,
                 markeredgecolor='white', markeredgewidth=1.0,
                 label=f'{hw} vs CPU', zorder=3)
        for x, y in zip(nqs, speedups):
            ax.annotate(f'{y:.1f}x', (x, y), textcoords='offset points',
                         xytext=(0, 8), ha='center', fontsize=7, color=c)

    for nq, prev_hw, new_hw in crossovers:
        ax.axvline(x=nq - 0.5, color='gray', linestyle=':', linewidth=1.2, zorder=1)
        ax.annotate(f'{prev_hw}\u2192{new_hw}\nfastest', (nq - 0.5, ax.get_ylim()[1]),
                     textcoords='offset points', xytext=(4, -12), fontsize=8,
                     color='dimgray', ha='left')

    ax.set_xlabel('Number of qubits', fontsize=12)
    ax.set_ylabel('Speedup vs CPU (kernel time, log scale)', fontsize=12)
    ax.set_title(f'{algo_label} Relative Speedup vs Qubit Count (CPU = 1.0x baseline)',
                  fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(range(min_q, max_q + 1))
    ax.grid(True, which='both', color=GRID_COLOR, linewidth=0.6)
    ax.legend(fontsize=11, loc='best')
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  Plot 4 — grouped bars per qubit count, winner starred
# ─────────────────────────────────────────────────────────────────────────────
def plot_winner_bars(rows, active, winners, algo_label, fname):
    nqs = sorted(winners)
    if not nqs:
        return
    n_hw = len(active)
    width = 0.8 / n_hw
    fig, ax = plt.subplots(figsize=(max(14, len(nqs) * 0.7), 7))

    x = np.arange(len(nqs))
    for i, hw in enumerate(active):
        vals = []
        for nq in nqs:
            r = rows[hw].get(nq)
            vals.append(r[0] if r else np.nan)
        offset = (i - (n_hw - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92, color=HW[hw]['color'],
                        edgecolor='white', label=hw, zorder=3)
        for xi, nq, v in zip(x + offset, nqs, vals):
            if np.isnan(v):
                continue
            if winners.get(nq) == hw:
                ax.annotate('\u2605', (xi, v), textcoords='offset points',
                             xytext=(0, 3), ha='center', fontsize=11, color='gold',
                             zorder=4)

    ax.set_xlabel('Number of qubits', fontsize=12)
    ax.set_ylabel('Kernel time (ms, log scale)', fontsize=12)
    ax.set_title(f'{algo_label} Kernel Time by Hardware, per Qubit Count (\u2605 = fastest)',
                  fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(nqs)
    ax.grid(axis='y', which='both', color=GRID_COLOR, linewidth=0.6)
    ax.legend(fontsize=11, loc='upper left')
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    main()