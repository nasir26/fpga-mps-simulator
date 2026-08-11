# #!/usr/bin/env python3
# """
# mcx_test_v07.py
# ================
# Standalone correctness + scaling test for the 'mcx' (multi-controlled-X)
# gate as used by the Grover oracle/diffusion ladder in
# comprehensive_test_v07.py (_ccz_ladder / _grover_diffusion delegate to
# 'mcx' for n >= 4 controls+target).

# This does NOT test Grover's algorithm itself — it isolates the MCX
# decomposition (Qiskit-backed, ancilla-free `mcx(mode='noancilla')`) and
# checks it fires/doesn't-fire correctly across 2–25 total qubits
# (1–24 controls + 1 target).

# Per qubit count n, three checks are run on an n-qubit register
# (controls = qubits[0..n-2], target = qubit[n-1]):

#   A. FIRE      — all controls set to |1>, single MCX applied.
#                  Target must flip 0->1.  Expect |1...1>.
#   B. NO-FIRE   — all controls except the last one set to |1>.
#                  MCX must NOT fire.  Target stays |0>.
#   C. DOUBLE    — all controls set to |1>, MCX applied twice (self-inverse).
#                  Target must end back at |0>. Expect same state as prep.

# Pass criterion
# --------------
#   Norm within 1e-5 of 1.0, AND the deterministic target index must carry
#   essentially all the probability (det_tolerance below).

# Caveats
# -------
#   - Qubit counts above ~15 mean statevectors of 2^n complex128 amplitudes
#     (2^25 -> ~512 MB just for the state vector), and the ancilla-free MCX
#     decomposition is O(n^2) gates, so wall time grows fast. Expect this
#     script to take a long time (potentially many minutes) to reach n=25 —
#     trim MAX_QUBITS below if you just need a quicker sanity sweep.
#   - The NO-FIRE check only tests a single missing-control position (the
#     last control). That is sufficient to catch the classic bug where a
#     MCX decomposition mis-handles the "not all controls satisfied" case,
#     but is not an exhaustive check of every control pattern.
# """

# import math, sys, time, traceback
# import numpy as np
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# try:
#     from fpga_simulator_v07 import (
#         QuantumFPGACircuit, FPGAValidator,
#         KERNEL_CLOCK_MHZ, NUM_HBM_BANKS, BYTES_PER_AMP, VERSION,
#     )
# except ImportError as e:
#     print(f"[FATAL] Cannot import fpga_simulator_v07: {e}")
#     sys.exit(1)

# XCLBIN       = "quantum_fpga_kernel_complex128.xclbin"
# SHOTS        = 0          # statevector-only
# NORM_TOL     = 1e-5
# DET_TOL      = 0.02       # probability tolerance on the deterministic index

# MIN_QUBITS = 2
# MAX_QUBITS = 25            # 1..24 controls + 1 target


# # ─────────────────────────────────────────────────────────────────────────────
# #  Result bookkeeping
# # ─────────────────────────────────────────────────────────────────────────────
# class R:
#     __slots__ = ('name', 'passed', 'norm', 'wall_us', 'detail')
#     def __init__(self, name, passed, norm, wall_us, detail=''):
#         self.name    = name
#         self.passed  = passed
#         self.norm    = norm
#         self.wall_us = wall_us
#         self.detail  = detail

#     def __str__(self):
#         s  = 'PASS' if self.passed else 'FAIL'
#         ms = self.wall_us / 1e3
#         d  = f'  [{self.detail}]' if self.detail else ''
#         return f"  [{s}] {self.name:<40s} norm={self.norm:.10f}  {ms:10.2f} ms{d}"


# _all_results = []


# def _prob(sv, idx):
#     return float(sv[idx].real**2 + sv[idx].imag**2)


# def _run_det(fpga, name, nq, gates, det_idx):
#     """Run a circuit and check norm + probability concentrated at det_idx."""
#     try:
#         t0 = time.perf_counter_ns()
#         res = fpga.run_circuit(
#             nq, gates, shots=SHOTS, validate=True,
#             det_idx=det_idx, det_tolerance=DET_TOL,
#             circuit_name=name)
#         wall_us = (time.perf_counter_ns() - t0) / 1e3
#         sv   = res['statevector']
#         val  = res['validation']
#         norm = val.norm if val else 0.0
#         norm_ok = abs(norm - 1.0) < NORM_TOL
#         p_target = _prob(sv, det_idx)
#         ok = norm_ok and (p_target >= 1.0 - DET_TOL)
#         kus = res.get('timing', {}).get('kernel_us', 0.0)
#         r = R(name, ok, norm, wall_us, f"P(target)={p_target:.6f}")
#         return r, wall_us, kus, p_target
#     except Exception as exc:
#         r = R(name, False, 0.0, 0.0, str(exc)[:80])
#         traceback.print_exc()
#         return r, 0.0, 0.0, 0.0


# # ═════════════════════════════════════════════════════════════════════════════
# #  MCX correctness sweep, 2–25 qubits
# # ═════════════════════════════════════════════════════════════════════════════
# _mcx_results = []   # (nq, test_label, passed, wall_us, kernel_us, prob)


# def run_mcx_sweep(fpga):
#     print('=' * 70)
#     print(f"  MCX (multi-controlled-X) Correctness Sweep "
#           f"({MIN_QUBITS}-{MAX_QUBITS} qubits)")
#     print('=' * 70)
#     print("  controls = qubits[0..n-2], target = qubit[n-1]")

#     for nq in range(MIN_QUBITS, MAX_QUBITS + 1):
#         controls = list(range(nq - 1))
#         target   = nq - 1
#         all_ones_idx    = (1 << nq) - 1                 # every bit = 1
#         target_zero_idx = all_ones_idx ^ (1 << target)  # controls=1, target=0

#         # ── A. FIRE: all controls=1, single MCX -> target flips 0->1 ──────
#         gates_fire = [('x', [c], None) for c in controls] + \
#                      [('mcx', controls + [target], None)]
#         name_a = f"MCX {nq:2d}q ({len(controls)} ctrl): FIRE all-ones"
#         r, wus, kus, p = _run_det(fpga, name_a, nq, gates_fire, all_ones_idx)
#         _all_results.append(r); _mcx_results.append((nq, 'FIRE', r.passed, wus, kus, p))
#         print(r)

#         # ── B. NO-FIRE: last control left at 0, MCX must not fire ─────────
#         if controls:
#             missing = controls[-1]
#             partial_controls = [c for c in controls if c != missing]
#             gates_nofire = [('x', [c], None) for c in partial_controls] + \
#                            [('mcx', controls + [target], None)]
#             # expected: all controls-minus-missing set, missing=0, target=0
#             expect_idx = 0
#             for c in partial_controls:
#                 expect_idx |= (1 << c)
#             name_b = f"MCX {nq:2d}q ({len(controls)} ctrl): NO-FIRE (ctrl{missing}=0)"
#             r, wus, kus, p = _run_det(fpga, name_b, nq, gates_nofire, expect_idx)
#             _all_results.append(r); _mcx_results.append((nq, 'NO-FIRE', r.passed, wus, kus, p))
#             print(r)

#         # ── C. DOUBLE: MCX applied twice == identity ───────────────────────
#         gates_double = [('x', [c], None) for c in controls] + \
#                        [('mcx', controls + [target], None)] * 2
#         name_c = f"MCX {nq:2d}q ({len(controls)} ctrl): DOUBLE=identity"
#         r, wus, kus, p = _run_det(fpga, name_c, nq, gates_double, target_zero_idx)
#         _all_results.append(r); _mcx_results.append((nq, 'DOUBLE', r.passed, wus, kus, p))
#         print(r)


# # ═════════════════════════════════════════════════════════════════════════════
# #  Visualisation
# # ═════════════════════════════════════════════════════════════════════════════
# _COLORS = {'FIRE': '#2ca02c', 'NO-FIRE': '#1f77b4', 'DOUBLE': '#ff7f0e',
#            'fail': '#d62728', 'grid': '#dddddd'}


# def _save(fig, path):
#     fig.tight_layout()
#     fig.savefig(path, dpi=150, bbox_inches='tight')
#     plt.close(fig)
#     print(f"  Saved -> {path}")


# def plot_mcx_sweep():
#     if not _mcx_results:
#         return

#     fig, axes = plt.subplots(1, 2, figsize=(14, 5))

#     # wall time by test type, colored, marked pass/fail with edge
#     ax = axes[0]
#     for label in ('FIRE', 'NO-FIRE', 'DOUBLE'):
#         pts = [(nq, wus/1e3) for (nq, lbl, ok, wus, kus, p) in _mcx_results if lbl == label]
#         if not pts:
#             continue
#         xs, ys = zip(*pts)
#         ax.plot(xs, ys, marker='o', label=label, color=_COLORS[label], linewidth=2)
#     ax.set_xlabel('Number of qubits (n)')
#     ax.set_ylabel('Wall time (ms)')
#     ax.set_title('MCX Correctness — Wall Time vs Qubits')
#     ax.set_yscale('log')
#     ax.grid(color=_COLORS['grid'])
#     ax.legend(fontsize=9)

#     # pass/fail scatter
#     ax = axes[1]
#     xs = list(range(len(_mcx_results)))
#     ys = [t[0] for t in _mcx_results]
#     cs = ['#2ca02c' if t[2] else _COLORS['fail'] for t in _mcx_results]
#     ax.scatter(xs, ys, c=cs, s=20, zorder=3)
#     ax.set_xlabel('Test index')
#     ax.set_ylabel('Number of qubits (n)')
#     ax.set_title('MCX Correctness — PASS (green) / FAIL (red)')
#     ax.grid(color=_COLORS['grid'], linewidth=0.5)

#     fig.suptitle(f'MCX Gate Correctness Sweep — FPGA v{VERSION} (Alveo U55C)',
#                  fontsize=13, fontweight='bold')
#     _save(fig, 'mcx_v07_sweep.png')


# # ═════════════════════════════════════════════════════════════════════════════
# #  MAIN
# # ═════════════════════════════════════════════════════════════════════════════
# def main():
#     print('=' * 70)
#     print(f'  FPGA Quantum Simulator v{VERSION} — MCX-Only Correctness Test')
#     print(f'  XCLBIN : {XCLBIN}')
#     print(f'  Clock  : {KERNEL_CLOCK_MHZ} MHz  |  HBM banks : {NUM_HBM_BANKS}')
#     print(f'  Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
#     print('=' * 70)

#     print('\nInitialising FPGA …')
#     t_init = time.perf_counter_ns()
#     try:
#         fpga = QuantumFPGACircuit(XCLBIN)
#     except Exception as e:
#         print(f'[FATAL] FPGA init: {e}')
#         sys.exit(1)
#     print(f'  Ready in {(time.perf_counter_ns()-t_init)/1e6:.1f} ms\n')

#     t_suite = time.perf_counter_ns()
#     run_mcx_sweep(fpga)
#     plot_mcx_sweep()
#     suite_s = (time.perf_counter_ns() - t_suite) / 1e9

#     total  = len(_all_results)
#     passed = sum(1 for r in _all_results if r.passed)
#     failed = total - passed

#     print(f'\n{"="*70}')
#     print(f'  FINAL RESULT : {passed}/{total} PASSED  ({failed} failed)')
#     print(f'  Suite time   : {suite_s:.1f} s')

#     if failed:
#         print(f'\n  Failed tests:')
#         for r in _all_results:
#             if not r.passed:
#                 print(f'    x {r.name}')

#     print(f'\n  Graph saved: mcx_v07_sweep.png')

#     status = 'ALL PASS' if not failed else f'{failed} FAILURES'
#     print(f'\n  Status : {status}')
#     print(f'{"="*70}\n')
#     sys.exit(0 if not failed else 1)


# if __name__ == '__main__':
#     main()


#!/usr/bin/env python3
"""
mcx_stress_test_v07.py
=======================
Extreme MCX (multi-controlled-X) correctness stress test, 2-15 qubits.

WHAT THIS SCRIPT DOES NOT DO
-----------------------------
This script never builds a Toffoli/MCX out of smaller gates itself. Every
MCX invocation is a single gate tuple:

        ('mcx', controls + [target], None)

fed straight into `fpga.run_circuit(...)`. Whatever internal algorithm
fpga_simulator_v07.py uses to realize that gate on hardware -- Gray-code
ladder, V-chain with ancilla, relative-phase Toffoli chain, direct native
synthesis, or anything else -- is entirely the simulator's business. This
file only:
  1. prepares an input basis state with plain X gates,
  2. calls 'mcx' once (or twice, for the self-inverse check),
  3. computes the CLASSICALLY-obvious expected answer (AND of the control
     bits decides whether the target flips) using plain Python integer/
     bitwise arithmetic -- not quantum gates, not a circuit, not a
     decomposition -- and checks the returned statevector against it.

If fpga_simulator_v07.py exposes more than one MCX-family gate name
(e.g. an explicit 'mcx_gray' or 'mcx_vchain' alongside 'mcx'), add those
names to GATE_NAMES_TO_TEST below and this script will sweep all of them
identically. Without seeing that file's ALL_SUPPORTED_GATES list, only
'mcx' is assumed to exist.

Test strategy per qubit count n (controls = qubits[0..n-2], target = qubit[n-1])
---------------------------------------------------------------------------------
  - n <= EXHAUSTIVE_MAX_QUBITS : every one of the 2^(n-1) control bit
    patterns is tested (full truth table).
  - n >  EXHAUSTIVE_MAX_QUBITS : a fixed set of deterministic edge
    patterns (all-zero, all-one, single-bit-set at each end, all-ones-
    minus-one at each end, alternating) PLUS a random sample of
    RANDOM_SAMPLES_PER_LEVEL additional patterns (seeded, reproducible).
  - One PHASE check per n: controls prepared |1...1>, target prepared in
    |+> = H|0>. A correctly-implemented MCX must act as identity on |+>
    when fired (X|+> = |+>), so applying H again must return the target
    deterministically to |0>. This catches phase/sign bugs that pure
    computational-basis flip tests (above) cannot see.
  - One DOUBLE-application (self-inverse) check per n on the all-ones
    pattern: MCX MCX == identity.

Resilience
----------
Because circuit cost grows fast with n, this script:
  - writes results incrementally to mcx_stress_results.csv after every
    completed test (not just at the end), so Ctrl+C / a crash never
    loses already-collected data,
  - catches KeyboardInterrupt and prints a partial summary instead of a
    raw traceback,
  - prints a running pass/fail tally as it goes.

Tune EXHAUSTIVE_MAX_QUBITS / RANDOM_SAMPLES_PER_LEVEL / MAX_QUBITS below
if a given run is taking too long -- MCX circuit cost does not depend
only on n, it also varies with the specific control pattern in ways that
are hard to predict in advance, so treat wall-time as unpredictable and
plan for this script to potentially run a long while at n>=12.
"""

import csv, itertools, math, os, random, sys, time, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from fpga_simulator_v07 import (
        QuantumFPGACircuit, FPGAValidator,
        KERNEL_CLOCK_MHZ, NUM_HBM_BANKS, BYTES_PER_AMP, VERSION,
    )
except ImportError as e:
    print(f"[FATAL] Cannot import fpga_simulator_v07: {e}")
    sys.exit(1)

XCLBIN   = "quantum_fpga_kernel_complex128.xclbin"
SHOTS    = 0
NORM_TOL = 1e-5
DET_TOL  = 0.02

MIN_QUBITS = 2
MAX_QUBITS = 15

EXHAUSTIVE_MAX_QUBITS    = 8    # n <= this: full 2^(n-1) truth table
RANDOM_SAMPLES_PER_LEVEL = 24   # n > this: random patterns on top of edge cases
RNG_SEED                 = 20260731

GATE_NAMES_TO_TEST = ['mcx']    # add e.g. 'mcx_gray' here if the simulator exposes it

CSV_PATH = 'mcx_stress_results.csv'

_rng = random.Random(RNG_SEED)


# ─────────────────────────────────────────────────────────────────────────────
#  Result bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
class R:
    __slots__ = ('name', 'passed', 'norm', 'wall_us', 'detail')
    def __init__(self, name, passed, norm, wall_us, detail=''):
        self.name    = name
        self.passed  = passed
        self.norm    = norm
        self.wall_us = wall_us
        self.detail  = detail

    def __str__(self):
        s  = 'PASS' if self.passed else 'FAIL'
        ms = self.wall_us / 1e3
        d  = f'  [{self.detail}]' if self.detail else ''
        return f"  [{s}] {self.name:<52s} norm={self.norm:.8f}  {ms:10.2f} ms{d}"


_all_results = []
_csv_rows    = []


def _prob(sv, idx):
    return float(sv[idx].real**2 + sv[idx].imag**2)


def _init_csv():
    with open(CSV_PATH, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['n_qubits', 'gate', 'test_type', 'pattern_bits',
                     'expected_idx', 'passed', 'norm', 'prob_target',
                     'wall_us', 'kernel_us', 'detail'])


def _append_csv(row):
    with open(CSV_PATH, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow(row)


def _run_det(fpga, name, nq, gates, det_idx):
    """Run a circuit; check norm + probability concentrated at det_idx.
    Purely a measurement of the simulator's output -- no gate math here."""
    try:
        t0 = time.perf_counter_ns()
        res = fpga.run_circuit(
            nq, gates, shots=SHOTS, validate=True,
            det_idx=det_idx, det_tolerance=DET_TOL,
            circuit_name=name)
        wall_us = (time.perf_counter_ns() - t0) / 1e3
        sv   = res['statevector']
        val  = res['validation']
        norm = val.norm if val else 0.0
        norm_ok = abs(norm - 1.0) < NORM_TOL
        p_target = _prob(sv, det_idx)
        ok = norm_ok and (p_target >= 1.0 - DET_TOL)
        kus = res.get('timing', {}).get('kernel_us', 0.0)
        r = R(name, ok, norm, wall_us, f"P(target)={p_target:.6f}")
        return r, wall_us, kus, p_target
    except Exception as exc:
        r = R(name, False, 0.0, 0.0, str(exc)[:80])
        traceback.print_exc()
        return r, 0.0, 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Classical expected-answer computation (bitwise arithmetic, NOT a circuit)
# ─────────────────────────────────────────────────────────────────────────────
def _expected_flip_index(pattern_bits, controls, target):
    """Given control bit values (tuple of 0/1, one per control qubit) and
    a target starting at 0, compute the basis-state index MCX should
    produce. This is plain classical logic (AND of control bits decides
    whether target flips) -- it is not a quantum decomposition and does
    not touch the FPGA."""
    idx = 0
    for c, bit in zip(controls, pattern_bits):
        if bit:
            idx |= (1 << c)
    fires = all(pattern_bits)
    if fires:
        idx |= (1 << target)
    return idx, fires


def _edge_patterns(n_controls):
    """Deterministic edge-case control patterns for stress coverage."""
    if n_controls == 0:
        return [()]
    pats = set()
    pats.add(tuple([0] * n_controls))                       # all zero
    pats.add(tuple([1] * n_controls))                        # all one
    first_only = [0] * n_controls; first_only[0] = 1
    pats.add(tuple(first_only))                               # only first
    last_only = [0] * n_controls; last_only[-1] = 1
    pats.add(tuple(last_only))                                 # only last
    all_but_first = [1] * n_controls; all_but_first[0] = 0
    pats.add(tuple(all_but_first))                              # all-but-first
    all_but_last = [1] * n_controls; all_but_last[-1] = 0
    pats.add(tuple(all_but_last))                                # all-but-last
    alt1 = tuple(i % 2 for i in range(n_controls))               # 0101...
    alt2 = tuple((i + 1) % 2 for i in range(n_controls))         # 1010...
    pats.add(alt1); pats.add(alt2)
    if n_controls >= 3:
        mid = [0] * n_controls; mid[n_controls // 2] = 1
        pats.add(tuple(mid))                                       # only middle
    return list(pats)


def _patterns_for_level(n_controls):
    """Full truth table for small n, edge+random sample for large n."""
    n_qubits = n_controls + 1
    if n_qubits <= EXHAUSTIVE_MAX_QUBITS:
        return list(itertools.product([0, 1], repeat=n_controls)), 'exhaustive'
    edges = _edge_patterns(n_controls)
    seen = set(edges)
    extra = []
    attempts = 0
    while len(extra) < RANDOM_SAMPLES_PER_LEVEL and attempts < RANDOM_SAMPLES_PER_LEVEL * 20:
        p = tuple(_rng.randint(0, 1) for _ in range(n_controls))
        attempts += 1
        if p in seen:
            continue
        seen.add(p)
        extra.append(p)
    return edges + extra, 'edge+random'


# ═════════════════════════════════════════════════════════════════════════════
#  Stress sweep
# ═════════════════════════════════════════════════════════════════════════════
def run_stress_sweep(fpga):
    print('=' * 74)
    print(f"  MCX EXTREME STRESS TEST  ({MIN_QUBITS}-{MAX_QUBITS} qubits)")
    print('=' * 74)
    print(f"  Exhaustive truth-table up to {EXHAUSTIVE_MAX_QUBITS}q, "
          f"edge+{RANDOM_SAMPLES_PER_LEVEL} random patterns above that.")
    print(f"  Gate(s) under test: {GATE_NAMES_TO_TEST}")
    print(f"  Results streamed to: {CSV_PATH}\n")

    running_pass = 0
    running_total = 0

    try:
        for nq in range(MIN_QUBITS, MAX_QUBITS + 1):
            controls = list(range(nq - 1))
            target   = nq - 1
            patterns, mode = _patterns_for_level(len(controls))

            print(f"--- n={nq:2d}q  ({len(controls)} controls)  "
                  f"[{mode}, {len(patterns)} patterns] ---")

            for gate_name in GATE_NAMES_TO_TEST:
                for pattern in patterns:
                    exp_idx, fires = _expected_flip_index(pattern, controls, target)
                    prep = [('x', [c], None) for c, b in zip(controls, pattern) if b]
                    gates = prep + [(gate_name, controls + [target], None)]
                    label = ''.join(str(b) for b in pattern) if pattern else '-'
                    name = (f"MCX[{gate_name}] {nq:2d}q ctrl={label} "
                            f"{'FIRE' if fires else 'no-fire'}")
                    r, wus, kus, p = _run_det(fpga, name, nq, gates, exp_idx)
                    _all_results.append(r)
                    running_total += 1
                    running_pass += int(r.passed)
                    row = [nq, gate_name, 'FIRE' if fires else 'NO-FIRE', label,
                           exp_idx, r.passed, r.norm, p, wus, kus, r.detail]
                    _append_csv(row)
                    print(f"{r}   [{running_pass}/{running_total} so far]")

                # self-inverse (double application) on the all-ones pattern
                all_ones = tuple([1] * len(controls))
                base_idx, _ = _expected_flip_index(all_ones, controls, target)  # target=1 case
                zero_idx = base_idx & ~(1 << target)  # after double-flip, target back to 0
                prep = [('x', [c], None) for c in controls]
                gates = prep + [(gate_name, controls + [target], None)] * 2
                name = f"MCX[{gate_name}] {nq:2d}q DOUBLE=identity (all-ones)"
                r, wus, kus, p = _run_det(fpga, name, nq, gates, zero_idx)
                _all_results.append(r)
                running_total += 1
                running_pass += int(r.passed)
                _append_csv([nq, gate_name, 'DOUBLE', '1'*len(controls),
                             zero_idx, r.passed, r.norm, p, wus, kus, r.detail])
                print(f"{r}   [{running_pass}/{running_total} so far]")

                # phase check: controls=1..1, target=|+>, MCX must act as
                # identity on |+> (X eigenstate) -- catches sign/phase bugs
                # that pure basis-flip tests can't see.
                prep = [('x', [c], None) for c in controls] + [('h', [target], None)]
                gates = prep + [(gate_name, controls + [target], None), ('h', [target], None)]
                zero_target_idx = 0
                for c in controls:
                    zero_target_idx |= (1 << c)  # controls=1, target=0
                name = f"MCX[{gate_name}] {nq:2d}q PHASE (|+> identity check)"
                r, wus, kus, p = _run_det(fpga, name, nq, gates, zero_target_idx)
                _all_results.append(r)
                running_total += 1
                running_pass += int(r.passed)
                _append_csv([nq, gate_name, 'PHASE', '1'*len(controls),
                             zero_target_idx, r.passed, r.norm, p, wus, kus, r.detail])
                print(f"{r}   [{running_pass}/{running_total} so far]")

            print()
    except KeyboardInterrupt:
        print(f"\n[Interrupted] Completed {running_total} tests "
              f"({running_pass} passed) before Ctrl+C. "
              f"Partial results are in {CSV_PATH}.")
        _print_summary(interrupted=True)
        sys.exit(130)


# ═════════════════════════════════════════════════════════════════════════════
#  Visualisation
# ═════════════════════════════════════════════════════════════════════════════
def plot_stress_sweep():
    if not _all_results:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    xs = list(range(len(_all_results)))
    ys = [r.wall_us / 1e3 for r in _all_results]
    cs = ['#2ca02c' if r.passed else '#d62728' for r in _all_results]
    ax.scatter(xs, ys, c=cs, s=10, zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel('Test index (chronological)')
    ax.set_ylabel('Wall time (ms)')
    ax.set_title(f'MCX Extreme Stress Test — All {len(_all_results)} Runs '
                 f'(FPGA v{VERSION})')
    ax.grid(color='#dddddd', linewidth=0.5)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#2ca02c',ms=8,label='PASS'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor='#d62728',ms=8,label='FAIL'),
    ], fontsize=10)
    fig.tight_layout()
    fig.savefig('mcx_stress_v07.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved -> mcx_stress_v07.png")


# ═════════════════════════════════════════════════════════════════════════════
#  Summary
# ═════════════════════════════════════════════════════════════════════════════
def _print_summary(interrupted=False):
    total  = len(_all_results)
    passed = sum(1 for r in _all_results if r.passed)
    failed = total - passed
    print(f'\n{"="*74}')
    print(f'  {"PARTIAL " if interrupted else ""}RESULT : {passed}/{total} PASSED  ({failed} failed)')
    if failed:
        print(f'\n  Failed tests:')
        for r in _all_results:
            if not r.passed:
                print(f'    x {r.name}  [{r.detail}]')
    print(f'  Full per-test data: {CSV_PATH}')
    print(f'{"="*74}\n')


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 74)
    print(f'  FPGA Quantum Simulator v{VERSION} — MCX EXTREME STRESS TEST')
    print(f'  XCLBIN : {XCLBIN}')
    print(f'  Clock  : {KERNEL_CLOCK_MHZ} MHz  |  HBM banks : {NUM_HBM_BANKS}')
    print(f'  Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 74)

    _init_csv()

    print('\nInitialising FPGA …')
    t_init = time.perf_counter_ns()
    try:
        fpga = QuantumFPGACircuit(XCLBIN)
    except Exception as e:
        print(f'[FATAL] FPGA init: {e}')
        sys.exit(1)
    print(f'  Ready in {(time.perf_counter_ns()-t_init)/1e6:.1f} ms\n')

    t_suite = time.perf_counter_ns()
    run_stress_sweep(fpga)
    plot_stress_sweep()
    suite_s = (time.perf_counter_ns() - t_suite) / 1e9

    print(f'  Suite time   : {suite_s:.1f} s')
    _print_summary()

    failed = sum(1 for r in _all_results if not r.passed)
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()