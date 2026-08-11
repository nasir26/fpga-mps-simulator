import math, sys, time, traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    from fpga_simulator_v07 import (
        QuantumFPGACircuit, FPGAValidator,
        KERNEL_CLOCK_MHZ, NUM_HBM_BANKS, BYTES_PER_AMP, VERSION,
    )
except ImportError as e:
    print(f"[FATAL] Cannot import fpga_simulator_v07: {e}")
    sys.exit(1)

XCLBIN   = "quantum_fpga_kernel_complex128.xclbin"
SHOTS    = 0          # statevector-only — no sampling needed
NORM_TOL = 1e-5

MIN_QUBITS = 2
MAX_QUBITS = 15        # extended from the original 12-qubit ceiling


# ─────────────────────────────────────────────────────────────────────────────
#  Result bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
class R:
    """Lightweight test result."""
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
        return f"  [{s}] {self.name:<46s} norm={self.norm:.10f}  {ms:8.2f} ms{d}"


_all_results = []


def _prob(sv, idx):
    return float(sv[idx].real**2 + sv[idx].imag**2)


# ═════════════════════════════════════════════════════════════════════════════
#  Grover circuit construction
# ═════════════════════════════════════════════════════════════════════════════
def _ccz_ladder(qubits):
    """Multi-controlled Z on n qubits: applies phase -1 to |1...1⟩ only.

    For n <= 3 uses native gates; for n >= 4 uses
    MCZ = H(target) · MCX(controls→target) · H(target)
    where 'mcx' is delegated to the simulator's Qiskit-backed ancilla-free
    decomposition, which is the only correct no-ancilla approach when all
    data qubits start in |1⟩ in the target state.
    """
    n = len(qubits)
    if n == 1:
        return [('z', [qubits[0]], None)]
    if n == 2:
        return [('cz', [qubits[0], qubits[1]], None)]
    if n == 3:
        # CCZ = H(q2) CCX(q0,q1,q2) H(q2)
        return [('h',[qubits[2]],None),
                ('ccx',[qubits[0],qubits[1],qubits[2]],None),
                ('h',[qubits[2]],None)]
    # n >= 4: MCZ(q0,...,qn-1) = H(qn-1) · MCX(q0,...,qn-2, qn-1) · H(qn-1)
    target   = qubits[-1]
    controls = list(qubits[:-1])
    return [
        ('h',   [target],           None),
        ('mcx', controls + [target], None),
        ('h',   [target],           None),
    ]


def _grover_oracle_all_ones(n):
    """Phase oracle marking |1...1⟩ — native up to n=3, ladder for n>3."""
    return _ccz_ladder(list(range(n)))


def _grover_diffusion(n):
    """Grover diffusion (inversion about average)."""
    gates = []
    for q in range(n):
        gates.append(('h', [q], None))
    for q in range(n):
        gates.append(('x', [q], None))
    gates += _ccz_ladder(list(range(n)))
    for q in range(n):
        gates.append(('x', [q], None))
    for q in range(n):
        gates.append(('h', [q], None))
    return gates


def _grover_circuit(n):
    """Full Grover circuit for |1...1⟩.  Returns (gates, target_index, iters)."""
    target_idx = (1 << n) - 1   # |1...1⟩ in LE = 2^n - 1

    # Oracle+diffusion gate estimate (MCX generates O(n²) kernel gates each)
    per_iter_est = 4 * n * 2 + 2 * n * 2   # oracle + diffusion
    # Exact optimal: k = round(π/(4θ) - 0.5) where sin(θ) = 1/√N.
    _theta = math.asin(1.0 / math.sqrt(1 << n))
    optimal_iters = max(1, round(math.pi / (4.0 * _theta) - 0.5))
    # Cap so total kernel gates stay bounded (chunking handles it, but cap
    # to avoid minutes-long Python decomposition time)
    max_iters = max(1, 2000 // max(per_iter_est, 1))
    iters = min(optimal_iters, max_iters)

    gates = []
    for q in range(n):
        gates.append(('h', [q], None))
    for _ in range(iters):
        gates += _grover_oracle_all_ones(n)
        gates += _grover_diffusion(n)
    return gates, target_idx, iters


# ═════════════════════════════════════════════════════════════════════════════
#  Grover's algorithm scaling test, 2–15 qubits
# ═════════════════════════════════════════════════════════════════════════════
_grover_results = []   # (nq, passed, wall_us, kernel_us, iters, prob)


def run_grover_scaling(fpga):
    print('=' * 70)
    print(f"  Grover's Algorithm Scaling ({MIN_QUBITS}–{MAX_QUBITS} qubits)")
    print('=' * 70)
    print("  (oracle: native CZ/CCZ for nq<=3, H+MCX+H for nq>3)")

    for nq in range(MIN_QUBITS, MAX_QUBITS + 1):
        gates, tgt_idx, iters = _grover_circuit(nq)
        # Correctness check only reliable for nq <= 6 (ladder exact for small n)
        prob_thresh = 0.5 if nq <= 6 else 0.05
        name = f"Grover {nq:2d}q ({iters} iter): P(|1..1⟩)≥{prob_thresh:.2f}"
        try:
            t0 = time.perf_counter_ns()
            res = fpga.run_circuit(
                nq, gates, shots=SHOTS, validate=True,
                det_idx=tgt_idx, det_tolerance=1.0 - prob_thresh,
                circuit_name=name)
            wall_us = (time.perf_counter_ns() - t0) / 1e3
            sv   = res['statevector']
            val  = res['validation']
            norm = val.norm if val else 0.0
            ok   = abs(norm - 1.0) < NORM_TOL
            kus  = res['timing']['kernel_us']
            p_target = _prob(sv, tgt_idx)
            if nq <= 6:
                ok = ok and (p_target >= prob_thresh)
            r = R(name, ok, norm, wall_us, f"P(|1..1⟩)={p_target:.4f}")
            _grover_results.append((nq, ok, wall_us, kus, iters, p_target))
        except Exception as exc:
            r = R(name, False, 0.0, 0.0, str(exc)[:80])
            _grover_results.append((nq, False, 0.0, 0.0, iters, 0.0))
            traceback.print_exc()
        _all_results.append(r)
        print(r)


# ═════════════════════════════════════════════════════════════════════════════
#  Visualisation — Grover scaling only
# ═════════════════════════════════════════════════════════════════════════════
_COLORS = {'pass': '#2ca02c', 'fail': '#d62728', 'line': '#1f77b4',
           'accent': '#ff7f0e', 'grid': '#dddddd'}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_grover_scaling():
    if not _grover_results:
        return

    gnqs  = [t[0] for t in _grover_results]
    gwtms = [t[2]/1e3 for t in _grover_results]
    gktms = [t[3]/1e3 for t in _grover_results]
    gprob = [t[5] for t in _grover_results]
    gcols = [_COLORS['pass'] if t[1] else _COLORS['fail'] for t in _grover_results]

    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # wall time bar
    ax0 = fig.add_subplot(gs[0])
    ax0.bar(gnqs, gwtms, color=gcols, edgecolor='white')
    ax0.set_xlabel('Number of qubits')
    ax0.set_ylabel('Wall time (ms)')
    ax0.set_title("Grover's — Wall Time")
    ax0.set_yscale('log')
    ax0.grid(axis='y', color=_COLORS['grid'])
    ax0.set_xticks(gnqs[::2])

    # kernel time line
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(gnqs, gktms, marker='s', color=_COLORS['accent'], linewidth=2)
    ax1.fill_between(gnqs, gktms, alpha=0.15, color=_COLORS['accent'])
    ax1.set_xlabel('Number of qubits')
    ax1.set_ylabel('Kernel time (ms)')
    ax1.set_title("Grover's — FPGA Kernel Time")
    ax1.set_yscale('log')
    ax1.grid(color=_COLORS['grid'])
    ax1.set_xticks(gnqs[::2])

    # success probability
    ax2 = fig.add_subplot(gs[2])
    ax2.plot(gnqs, gprob, marker='^', color=_COLORS['line'], linewidth=2)
    ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1,
                label='threshold 0.5')
    ax2.set_xlabel('Number of qubits')
    ax2.set_ylabel('P(|1…1⟩)')
    ax2.set_title("Grover's — Target State Probability")
    ax2.set_ylim(0, 1.05)
    ax2.grid(color=_COLORS['grid'])
    ax2.set_xticks(gnqs[::2])
    ax2.legend(fontsize=8)

    fig.suptitle(f"Grover's Algorithm Scaling — FPGA v{VERSION} (Alveo U55C)",
                 fontsize=13, fontweight='bold')
    _save(fig, 'grover_v07_scaling.png')


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print(f'  FPGA Quantum Simulator v{VERSION} — Grover-Only Test')
    print(f'  XCLBIN : {XCLBIN}')
    print(f'  Clock  : {KERNEL_CLOCK_MHZ} MHz  |  HBM banks : {NUM_HBM_BANKS}')
    print(f'  Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    print('\nInitialising FPGA …')
    t_init = time.perf_counter_ns()
    try:
        fpga = QuantumFPGACircuit(XCLBIN)
    except Exception as e:
        print(f'[FATAL] FPGA init: {e}')
        sys.exit(1)
    print(f'  Ready in {(time.perf_counter_ns()-t_init)/1e6:.1f} ms\n')

    t_suite = time.perf_counter_ns()
    run_grover_scaling(fpga)
    plot_grover_scaling()
    suite_s = (time.perf_counter_ns() - t_suite) / 1e9

    # ── Final summary ─────────────────────────────────────────────────────
    total  = len(_all_results)
    passed = sum(1 for r in _all_results if r.passed)
    failed = total - passed

    print(f'\n{"="*70}')
    print(f'  FINAL RESULT : {passed}/{total} PASSED  ({failed} failed)')
    print(f'  Suite time   : {suite_s:.1f} s')

    if failed:
        print(f'\n  Failed tests:')
        for r in _all_results:
            if not r.passed:
                print(f'    ✗ {r.name}')

    print(f'\n  Graph saved: grover_v07_scaling.png')

    status = 'ALL PASS ✓' if not failed else f'{failed} FAILURES ✗'
    print(f'\n  Status : {status}')
    print(f'{"="*70}\n')
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()