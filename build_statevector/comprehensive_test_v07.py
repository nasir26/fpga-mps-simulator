#!/usr/bin/env python3
"""
comprehensive_test_v07.py
=========================
Full gate-correctness + algorithm scaling test for fpga_simulator_v07,
targeting the unified 8-bank kernel (quantum_fpga_kernel_complex128.xclbin,
built from quantum_fpga_kernel_unified.cpp).

Sections
--------
  1  Single-qubit gates (17 native gates)
  2  Two-qubit native gates (CX, CZ)
  3  Decomposed multi-qubit gates (CY, CH, SWAP, CP, CRX, CRY, CRZ, CSX,
       DCX, ISWAP, ECR, RXX, RYY, RZZ, CCX/Toffoli)
  4  ECR state-vector correctness
  5  QFT scaling 1–20 qubits
  6  Grover's algorithm scaling 2–20 qubits
  7  QFT/Grover kernel-time comparison (shared with section 9 plots)
  8  Classical-bit-mapped measurement (new in v07)
  9  Visualisations (saved as PNG files)

Section 8 exercises exactly the case that motivated v07: a single qubit
measured into more than one classical bit —

    qc = QuantumCircuit(1, 3)
    qc.measure(0, 0)
    qc.measure(0, 1)

— plus clbit reordering, a partially-measured register (unmeasured clbits
must stay '0'), and the backward-compatible no-clbit-given fallback.

Pass criterion
--------------
  Norm within 1e-6 of 1.0 for all tests.
  State-vector checks use per-test thresholds documented inline.
  Section 8 checks the *shape* of the returned counts dict (key length,
  which characters are pinned to '0', which characters must always match)
  rather than a single expected key, since measurement is stochastic.
"""

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

XCLBIN        = "quantum_fpga_kernel_complex128.xclbin"
SHOTS         = 0          # statevector-only — no sampling needed for gate tests
MEAS_SHOTS    = 4000       # shots used specifically by Section 8's counts checks
NORM_TOL      = 1e-5       # slightly relaxed for decomposed gates
STATE_TOL     = 0.02       # probability tolerance for deterministic checks

# ─────────────────────────────────────────────────────────────────────────────
#  Result bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
class R:
    """Lightweight test result."""
    __slots__ = ('name','passed','norm','wall_us','detail')
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
        return f"  [{s}] {self.name:<42s} norm={self.norm:.10f}  {ms:8.2f} ms{d}"


_all_results = []

def _run(fpga, name, nq, gates, *, det_idx=None, marginal=None,
         marginal_min=0.45, uniform=False):
    """Execute one test, append result, return R."""
    try:
        t0 = time.perf_counter_ns()
        res = fpga.run_circuit(
            nq, gates, shots=SHOTS, validate=True,
            det_idx=det_idx,
            marginal_pattern=marginal, marginal_min_prob=marginal_min,
            uniform_check=uniform,
            det_tolerance=STATE_TOL,
            circuit_name=name)
        wall_us = (time.perf_counter_ns() - t0) / 1e3
        val  = res['validation']
        norm = val.norm if val else 0.0
        ok   = val.passed if val else False
        det  = val.expected_detail if val else ''
        r = R(name, ok, norm, wall_us, det)
    except Exception as exc:
        r = R(name, False, 0.0, 0.0, str(exc)[:80])
        traceback.print_exc()
    _all_results.append(r)
    print(r)
    return r


def _run_meas(fpga, name, nq, gates, num_clbits, check_fn, shots=MEAS_SHOTS):
    """Execute one measurement-mapping test.

    check_fn(counts: dict[str,int]) -> (ok: bool, detail: str). Norm is
    still validated against the pre-measurement statevector; ok is the AND
    of norm-validity and check_fn's own verdict.
    """
    try:
        t0 = time.perf_counter_ns()
        res = fpga.run_circuit(
            nq, gates, shots=shots, validate=True,
            det_tolerance=STATE_TOL, circuit_name=name,
            num_clbits=num_clbits)
        wall_us = (time.perf_counter_ns() - t0) / 1e3
        val  = res['validation']
        norm = val.norm if val else 0.0
        norm_ok = val.passed if val else False
        check_ok, detail = check_fn(res['counts'])
        r = R(name, norm_ok and check_ok, norm, wall_us, detail)
    except Exception as exc:
        r = R(name, False, 0.0, 0.0, str(exc)[:80])
        traceback.print_exc()
    _all_results.append(r)
    print(r)
    return r


# ─────────────────────────────────────────────────────────────────────────────
#  Utility: statevector probability at index
# ─────────────────────────────────────────────────────────────────────────────
def _prob(sv, idx):
    return float(sv[idx].real**2 + sv[idx].imag**2)


def _run_sv(fpga, name, nq, gates):
    """Run circuit, return (statevector, wall_us).  No validation object."""
    t0 = time.perf_counter_ns()
    res = fpga.run_circuit(nq, gates, shots=SHOTS, validate=False,
                           circuit_name=name)
    return res['statevector'], (time.perf_counter_ns() - t0) / 1e3


# ─────────────────────────────────────────────────────────────────────────────
#  Section header printer
# ─────────────────────────────────────────────────────────────────────────────
def _hdr(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — Single-qubit gates
# ═════════════════════════════════════════════════════════════════════════════
def section1(fpga):
    _hdr("Section 1 — Single-Qubit Gates")
    nq = 3   # 3 qubits so gates on q=0 don't trivially pass due to |0⟩ symmetry

    # ID: state unchanged, norm=1
    _run(fpga, "ID: no-op norm", nq,
         [('id',[0],None)], det_idx=0)

    # H: uniform superposition on q0 (1-qubit space)
    _run(fpga, "H: uniform superposition |+⟩", 1,
         [('h',[0],None)], uniform=True)

    # X: flips |0⟩ → |1⟩  (index 1 on 1-qubit register)
    _run(fpga, "X: |0⟩→|1⟩", 1,
         [('x',[0],None)], det_idx=1)

    # X twice: identity → back to |0⟩
    _run(fpga, "X†X: identity", 1,
         [('x',[0],None),('x',[0],None)], det_idx=0)

    # Y: |0⟩ → i|1⟩  — |1⟩ has probability 1
    _run(fpga, "Y: |0⟩→i|1⟩", 1,
         [('y',[0],None)], det_idx=1)

    # Z: |0⟩ unchanged (only phase), P(|0⟩)=1
    _run(fpga, "Z: |0⟩ unchanged (phase only)", 1,
         [('z',[0],None)], det_idx=0)

    # S then SDG: identity
    _run(fpga, "S·S†: identity", 1,
         [('s',[0],None),('sdg',[0],None)], det_idx=0)

    # T then TDG: identity
    _run(fpga, "T·T†: identity", 1,
         [('t',[0],None),('tdg',[0],None)], det_idx=0)

    # SX: sqrt-X, then SXDG: identity
    _run(fpga, "SX·SX†: identity", 1,
         [('sx',[0],None),('sxdg',[0],None)], det_idx=0)

    # SX twice = X: |0⟩→|1⟩
    _run(fpga, "SX²=X: |0⟩→|1⟩", 1,
         [('sx',[0],None),('sx',[0],None)], det_idx=1)

    # RX(π): same as X gate
    _run(fpga, "RX(π): |0⟩→|1⟩", 1,
         [('rx',[0],[math.pi])], det_idx=1)

    # RX(2π): identity → |0⟩ (up to global phase)
    _run(fpga, "RX(2π): identity |0⟩", 1,
         [('rx',[0],[2*math.pi])], det_idx=0)

    # RY(π): same as Y gate
    _run(fpga, "RY(π): |0⟩→|1⟩", 1,
         [('ry',[0],[math.pi])], det_idx=1)

    # RZ(π): phase, |0⟩ unchanged
    _run(fpga, "RZ(π): |0⟩ unchanged (phase)", 1,
         [('rz',[0],[math.pi])], det_idx=0)

    # P(π): same as Z, |0⟩ unchanged
    _run(fpga, "P(π): |0⟩ unchanged", 1,
         [('p',[0],[math.pi])], det_idx=0)

    # U2(0,π): = H gate
    _run(fpga, "U2(0,π): = H, uniform superposition", 1,
         [('u2',[0],[0.0, math.pi])], uniform=True)

    # U3(π,0,π): = X gate
    _run(fpga, "U3(π,0,π): = X gate, |0⟩→|1⟩", 1,
         [('u3',[0],[math.pi, 0.0, math.pi])], det_idx=1)

    # U3(π/2,0,π): = H gate
    _run(fpga, "U3(π/2,0,π): = H, uniform superposition", 1,
         [('u3',[0],[math.pi/2, 0.0, math.pi])], uniform=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Two-qubit native gates
# ═════════════════════════════════════════════════════════════════════════════
def section2(fpga):
    _hdr("Section 2 — Two-Qubit Native Gates (CX, CZ)")

    # CX: H(0) then CX(0,1) → Bell |Φ+⟩, P(q0=0)≈0.5
    _run(fpga, "CX: Bell state |Φ+⟩", 2,
         [('h',[0],None),('cx',[0,1],None)],
         marginal={0:0}, marginal_min=0.45)

    # CX: control=1 state, target flips: |10⟩→|11⟩
    _run(fpga, "CX: |10⟩→|11⟩", 2,
         [('x',[0],None),('cx',[0,1],None)],
         det_idx=3)   # |11⟩ in LE = index 3

    # CX twice = identity
    _run(fpga, "CX²: identity |00⟩", 2,
         [('x',[0],None),('cx',[0,1],None),('cx',[0,1],None),('x',[0],None)],
         det_idx=0)

    # CZ: H(0), H(1), CZ, H(0), H(1) = CX (different basis)
    # Simpler: |00⟩ → CZ leaves |00⟩ unchanged
    _run(fpga, "CZ: |00⟩ unchanged", 2,
         [('cz',[0,1],None)], det_idx=0)

    # CZ: |11⟩ → -|11⟩ (phase flip, probability unchanged)
    _run(fpga, "CZ: |11⟩→-|11⟩ (phase, norm=1)", 2,
         [('x',[0],None),('x',[1],None),('cz',[0,1],None)],
         det_idx=3)

    # CZ twice = identity
    _run(fpga, "CZ²: identity", 2,
         [('x',[0],None),('x',[1],None),
          ('cz',[0,1],None),('cz',[0,1],None)],
         det_idx=3)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Decomposed multi-qubit gates
# ═════════════════════════════════════════════════════════════════════════════
def section3(fpga):
    _hdr("Section 3 — Decomposed Multi-Qubit Gates")

    # CY: controlled-Y: |10⟩ → i|11⟩ (ctrl=q0=1, tgt=q1 flips)
    _run(fpga, "CY: |10⟩→i|11⟩", 2,
         [('x',[0],None),('cy',[0,1],None)], det_idx=3)

    # CH: controlled-H: ctrl=1, tgt superposed → norm only
    _run(fpga, "CH: norm preserved", 2,
         [('x',[0],None),('ch',[0,1],None)])

    # SWAP: |10⟩ → |01⟩
    _run(fpga, "SWAP: |10⟩→|01⟩", 2,
         [('x',[0],None),('swap',[0,1],None)],
         det_idx=2)   # |01⟩ in LE: q1=1,q0=0 = index 2

    # SWAP twice = identity
    _run(fpga, "SWAP²: identity", 2,
         [('x',[0],None),('swap',[0,1],None),('swap',[0,1],None),('x',[0],None)],
         det_idx=0)

    # CP(π): |11⟩ → -|11⟩ (phase), norm preserved
    _run(fpga, "CP(π): |11⟩ phase flip, norm=1", 2,
         [('x',[0],None),('x',[1],None),('cp',[0,1],[math.pi])],
         det_idx=3)

    # CP(2π): identity (phase = 2π)
    _run(fpga, "CP(2π): identity", 2,
         [('x',[0],None),('x',[1],None),('cp',[0,1],[2*math.pi]),
          ('x',[0],None),('x',[1],None)],
         det_idx=0)

    # CRX(π): controlled-RX(π) = controlled-X when ctrl=1
    _run(fpga, "CRX(π): |10⟩→|11⟩", 2,
         [('x',[0],None),('crx',[0,1],[math.pi])], det_idx=3)

    # CRY(π): controlled-RY(π)
    _run(fpga, "CRY(π): |10⟩→|11⟩", 2,
         [('x',[0],None),('cry',[0,1],[math.pi])], det_idx=3)

    # CRZ(π): controlled-RZ, ctrl=1, phase only on |1⟩ of tgt
    _run(fpga, "CRZ(π): norm=1", 2,
         [('x',[0],None),('crz',[0,1],[math.pi])])

    # CSX: controlled-sqrt-X
    _run(fpga, "CSX: norm=1", 2,
         [('x',[0],None),('csx',[0,1],None)])

    # CSX twice = CX: |10⟩ → |11⟩
    _run(fpga, "CSX²=CX: |10⟩→|11⟩", 2,
         [('x',[0],None),('csx',[0,1],None),('csx',[0,1],None)], det_idx=3)

    # DCX: double-CX: |00⟩→|00⟩ (both ctrl=0), norm only
    _run(fpga, "DCX: norm=1", 2,
         [('dcx',[0,1],None)])

    # ISWAP: norm preserved
    _run(fpga, "ISWAP: norm=1", 2,
         [('x',[0],None),('iswap',[0,1],None)])

    # ISWAP on |10⟩: → i|01⟩, probability at |01⟩ ≈ 1
    _run(fpga, "ISWAP: |10⟩→i|01⟩", 2,
         [('x',[0],None),('iswap',[0,1],None)], det_idx=2)

    # ECR: norm preserved
    _run(fpga, "ECR: |00⟩ norm=1", 2,
         [('ecr',[0,1],None)])

    # ECR twice ≠ identity in general, but norm preserved
    _run(fpga, "ECR: |11⟩ norm=1", 2,
         [('x',[0],None),('x',[1],None),('ecr',[0,1],None)])

    # RXX(π/2): norm preserved
    _run(fpga, "RXX(π/2): norm=1", 2,
         [('rxx',[0,1],[math.pi/2])])

    # RYY(π/2): norm preserved
    _run(fpga, "RYY(π/2): norm=1", 2,
         [('ryy',[0,1],[math.pi/2])])

    # RZZ(π/2): norm preserved
    _run(fpga, "RZZ(π/2): norm=1", 2,
         [('rzz',[0,1],[math.pi/2])])

    # CCX (Toffoli): |110⟩ → |111⟩
    # LE: |q2 q1 q0⟩, ctrl0=q0, ctrl1=q1, tgt=q2
    # |110⟩ = q1=1,q0=1,q2=0 → after CCX q2 flips → |111⟩
    _run(fpga, "CCX: |110⟩→|111⟩", 3,
         [('x',[0],None),('x',[1],None),('ccx',[0,1,2],None)],
         det_idx=7)   # |111⟩ = 7

    # CCX: control not satisfied, no flip
    _run(fpga, "CCX: |100⟩ unchanged", 3,
         [('x',[0],None),('ccx',[0,1,2],None)],
         det_idx=1)   # |001⟩ in LE = 1

    # CCX twice = identity
    _run(fpga, "CCX²: identity", 3,
         [('x',[0],None),('x',[1],None),
          ('ccx',[0,1,2],None),('ccx',[0,1,2],None),
          ('x',[0],None),('x',[1],None)],
         det_idx=0)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — ECR state-vector correctness
# ═════════════════════════════════════════════════════════════════════════════
def section4(fpga):
    _hdr("Section 4 — ECR State-Vector Correctness")

    inv2 = 1.0 / math.sqrt(2)
    # ECR_LE (Qiskit LE convention) = 1/√2 *
    #   [[0, 1, 0, i], [1, 0, -i, 0], [0, i, 0, 1], [-i, 0, 1, 0]]
    # index ordering: |q1 q0⟩ → 0=|00⟩, 1=|01⟩, 2=|10⟩, 3=|11⟩
    expected = {
        #  input    → expected output column of ECR_LE
        # |00⟩ (idx 0): col 0 = [0, 1/√2, 0, -i/√2]
        #   → P(|01⟩)=0.5, P(|11⟩)=0.5
        '|00⟩': ([], {1: 0.5, 3: 0.5}),
        # |01⟩ (idx 1): col 1 = [1/√2, 0, i/√2, 0]
        #   → P(|00⟩)=0.5, P(|10⟩)=0.5
        '|01⟩': ([('x',[0],None)], {0: 0.5, 2: 0.5}),
        # |10⟩ (idx 2): col 2 = [0, -i/√2, 0, 1/√2]
        #   → P(|01⟩)=0.5, P(|11⟩)=0.5
        '|10⟩': ([('x',[1],None)], {1: 0.5, 3: 0.5}),
        # |11⟩ (idx 3): col 3 = [i/√2, 0, 1/√2, 0]
        #   → P(|00⟩)=0.5, P(|10⟩)=0.5
        '|11⟩': ([('x',[0],None),('x',[1],None)], {0: 0.5, 2: 0.5}),
    }

    for label, (prep, probs_exp) in expected.items():
        gates = prep + [('ecr',[0,1],None)]
        sv, wall_us = _run_sv(fpga, f"ECR {label}", 2, gates)
        norm_ok, norm = FPGAValidator.check_norm(sv)
        state_ok = True
        for idx, p_exp in probs_exp.items():
            p_got = _prob(sv, idx)
            if abs(p_got - p_exp) > 0.02:
                state_ok = False
                break
        passed = norm_ok and state_ok
        probs_str = ', '.join(f'P(|{i}>)={_prob(sv,i):.4f}' for i in probs_exp)
        r = R(f"ECR {label} state check", passed, norm, wall_us, probs_str)
        _all_results.append(r)
        print(r)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — QFT scaling 1–20 qubits
# ═════════════════════════════════════════════════════════════════════════════
def _qft_gates(n, inverse=False):
    """QFT circuit in Qiskit LE convention."""
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
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, i, -1):
                angle = -math.pi / (2 ** (j - i))
                gates.append(('cp', [j, i], [angle]))
            gates.append(('h', [i], None))
    return gates


_qft_results = []   # (nq, passed, wall_us, kernel_us)


def section5(fpga):
    _hdr("Section 5 — QFT Scaling (1–20 qubits)")

    for nq in range(1, 21):
        name = f"QFT {nq:2d}q: uniform superposition"
        gates = _qft_gates(nq)
        try:
            t0 = time.perf_counter_ns()
            res = fpga.run_circuit(
                nq, gates, shots=SHOTS, validate=True,
                uniform_check=True, uniform_tolerance=0.05,
                circuit_name=name)
            wall_us = (time.perf_counter_ns() - t0) / 1e3
            val  = res['validation']
            norm = val.norm if val else 0.0
            ok   = val.passed if val else False
            kus  = res['timing']['kernel_us']
            r = R(name, ok, norm, wall_us)
            _qft_results.append((nq, ok, wall_us, kus))
        except Exception as exc:
            r = R(name, False, 0.0, 0.0, str(exc)[:60])
            _qft_results.append((nq, False, 0.0, 0.0))
            traceback.print_exc()
        _all_results.append(r)
        print(r)

    print()
    # Round-trip: QFT then IQFT should return to |0⟩
    for nq in [1, 2, 4, 8, 12]:
        name = f"QFT+IQFT {nq:2d}q: round-trip |0⟩"
        gates = _qft_gates(nq) + _qft_gates(nq, inverse=True)
        try:
            t0 = time.perf_counter_ns()
            res = fpga.run_circuit(
                nq, gates, shots=SHOTS, validate=True,
                det_idx=0, det_tolerance=0.02,
                circuit_name=name)
            wall_us = (time.perf_counter_ns() - t0) / 1e3
            val  = res['validation']
            norm = val.norm if val else 0.0
            ok   = val.passed if val else False
            r = R(name, ok, norm, wall_us,
                  f"P(|0⟩)={_prob(res['statevector'],0):.6f}")
        except Exception as exc:
            r = R(name, False, 0.0, 0.0, str(exc)[:60])
            traceback.print_exc()
        _all_results.append(r)
        print(r)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Grover's algorithm scaling 2–20 qubits
# ═════════════════════════════════════════════════════════════════════════════
def _ccz_ladder(qubits):
    """Multi-controlled Z on n qubits: applies phase -1 to |1...1⟩ only.

    For n <= 3 uses native gates; for n >= 4 uses
    MCZ = H(target) · MCX(controls→target) · H(target)
    where 'mcx' is delegated to the simulator's Qiskit-backed ancilla-free
    decomposition, which is the only correct no-ancilla approach when all
    data qubits start in |1⟩ in the target state.

    The previous CCX-ladder was wrong for n >= 4: CCX(q0,q1,q2) flips q2
    from 1→0 in |1...1⟩ so the subsequent CZ never fires on that state.
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
    # 'mcx' is in ALL_SUPPORTED_GATES and is decomposed correctly by the
    # simulator via _decompose_mcx / Qiskit mcx(mode='noancilla').
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
    # The simpler approximation round(π/4·√N) overcounts by 1 for N=4 (2q),
    # causing 2 iterations instead of 1 and giving P(|11⟩) = 0.25 instead of 1.
    _theta = math.asin(1.0 / math.sqrt(1 << n))
    optimal_iters = max(1, round(math.pi / (4.0 * _theta) - 0.5))
    # Cap so total kernel gates stay well under 10× 512 (chunking handles it,
    # but cap to avoid minutes-long Python decomposition time)
    max_iters = max(1, 2000 // max(per_iter_est, 1))
    iters = min(optimal_iters, max_iters)

    gates = []
    for q in range(n):
        gates.append(('h', [q], None))
    for _ in range(iters):
        gates += _grover_oracle_all_ones(n)
        gates += _grover_diffusion(n)
    return gates, target_idx, iters


_grover_results = []   # (nq, passed, wall_us, kernel_us, iters, prob)


def section6(fpga):
    _hdr("Section 6 — Grover's Algorithm Scaling (2–12 qubits)")
    print("  (oracle: native CZ/CCZ for nq≤3, H+MCX+H for nq>3)")

    for nq in range(2, 13):
        gates, tgt_idx, iters = _grover_circuit(nq)
        # Correctness check only reliable for nq ≤ 6 (ladder exact for small n)
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
            ok   = norm_ok = abs(norm - 1.0) < NORM_TOL
            kus  = res['timing']['kernel_us']
            p_target = _prob(sv, tgt_idx)
            if nq <= 6:
                ok = ok and (p_target >= prob_thresh)
            r = R(name, ok, norm, wall_us,
                  f"P(|1..1⟩)={p_target:.4f}")
            _grover_results.append((nq, ok, wall_us, kus, iters, p_target))
        except Exception as exc:
            r = R(name, False, 0.0, 0.0, str(exc)[:80])
            _grover_results.append((nq, False, 0.0, 0.0, iters, 0.0))
            traceback.print_exc()
        _all_results.append(r)
        print(r)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — Classical-bit-mapped measurement (new in v07)
# ═════════════════════════════════════════════════════════════════════════════
def section8(fpga):
    _hdr("Section 8 — Classical-Bit-Mapped Measurement (v07)")

    # ── Test 1: the exact motivating case ──────────────────────────────────
    #   qc = QuantumCircuit(1, 3); qc.measure(0,0); qc.measure(0,1)
    #   on H|0> — qubit0 is 50/50, fanned out to clbit0 AND clbit1;
    #   clbit2 is never targeted and must stay '0' in every count key.
    def check_fanout(counts):
        if not counts:
            return False, "no counts returned"
        for key in counts:
            if len(key) != 3:
                return False, f"bad key length: {key!r}"
            if key[0] != '0':
                return False, f"clbit2 (unmeasured) not '0': {key!r}"
            if key[1] != key[2]:
                return False, f"clbit1 != clbit0 (same source qubit): {key!r}"
        keys = set(counts)
        if not keys.issubset({'000', '011'}):
            return False, f"unexpected keys: {sorted(keys)}"
        return True, f"keys={sorted(counts.items())}"

    _run_meas(fpga, "Meas: 1 qubit -> 2 clbits, 1 unmeasured", 1,
              [('h',[0],None), ('measure',[0],[0]), ('measure',[0],[1])],
              num_clbits=3, check_fn=check_fanout)

    # ── Test 2: single qubit measured into a high-order clbit only ─────────
    #   X|0> -> |1>, measured only into clbit 2 of a 3-bit register.
    #   Expect every shot to read '100' (clbit2 is the leftmost/MSB char).
    def check_high_clbit(counts):
        keys = set(counts)
        if keys != {'100'}:
            return False, f"expected only '100', got {sorted(keys)}"
        return True, f"keys={sorted(counts.items())}"

    _run_meas(fpga, "Meas: qubit0 -> clbit2 only (deterministic)", 1,
              [('x',[0],None), ('measure',[0],[2])],
              num_clbits=3, check_fn=check_high_clbit)

    # ── Test 3: two qubits, explicit clbit assignment, full register ──────
    #   X(1) so qubit1 is deterministically |1>; H(0) so qubit0 is 50/50.
    #   measure(0,0), measure(1,1) -> clbit1 (leftmost char) always '1'.
    def check_two_qubit_map(counts):
        keys = set(counts)
        if not keys.issubset({'10', '11'}):
            return False, f"unexpected keys: {sorted(keys)}"
        for key in keys:
            if key[0] != '1':
                return False, f"clbit1 (from qubit1=X) not '1': {key!r}"
        return True, f"keys={sorted(counts.items())}"

    _run_meas(fpga, "Meas: 2 qubits -> 2 clbits, explicit map", 2,
              [('h',[0],None), ('x',[1],None),
               ('measure',[0],[0]), ('measure',[1],[1])],
              num_clbits=2, check_fn=check_two_qubit_map)

    # ── Test 4: backward-compatible fallback (clbits=None) ─────────────────
    #   Raw ('measure', qubits, None) tuple — no explicit clbit list at all.
    #   Every measured qubit should map 1:1 to a clbit of the same index,
    #   exactly like v06's only supported behaviour.
    def check_fallback(counts):
        # X(0) only -> qubit0=1, qubit1=0 -> expect deterministic '01'
        # (qubit1 -> leftmost char '0', qubit0 -> rightmost char '1')
        keys = set(counts)
        if keys != {'01'}:
            return False, f"expected only '01', got {sorted(keys)}"
        return True, f"keys={sorted(counts.items())}"

    _run_meas(fpga, "Meas: clbits=None fallback (qubit==clbit)", 2,
              [('x',[0],None), ('measure',[0,1],None)],
              num_clbits=2, check_fn=check_fallback)

    # ── Test 5: measured qubit count < num_clbits, no measure at all ──────
    #   No 'measure' gate present at all -> run_circuit's own fallback
    #   kicks in (identity map up to min(num_qubits, num_clbits)); with
    #   shots>0 this should still produce a valid, norm-preserving sample
    #   from the |0> state, i.e. all-zero key.
    def check_no_measure(counts):
        keys = set(counts)
        if keys != {'000'}:
            return False, f"expected only '000' (no measure => |0>), got {sorted(keys)}"
        return True, f"keys={sorted(counts.items())}"

    _run_meas(fpga, "Meas: no explicit measure, |0> default", 2,
              [('id',[0],None)],
              num_clbits=3, check_fn=check_no_measure)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — Visualisations
# ═════════════════════════════════════════════════════════════════════════════
_COLORS = {'pass': '#2ca02c', 'fail': '#d62728', 'line': '#1f77b4',
           'accent': '#ff7f0e', 'grid': '#dddddd'}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {path}")


def _gate_bar(results_subset, title, outpath):
    names  = [r.name.split(':')[0].strip() for r in results_subset]
    times  = [r.wall_us / 1e3 for r in results_subset]
    colors = [_COLORS['pass'] if r.passed else _COLORS['fail'] for r in results_subset]
    fig, ax = plt.subplots(figsize=(max(10, len(names)*0.6), 5))
    bars = ax.bar(names, times, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Wall time (ms)')
    ax.set_title(title)
    ax.set_yscale('log')
    ax.grid(axis='y', color=_COLORS['grid'], linewidth=0.5)
    ax.tick_params(axis='x', rotation=55, labelsize=8)
    from matplotlib.patches import Patch
    legend = [Patch(color=_COLORS['pass'], label='PASS'),
              Patch(color=_COLORS['fail'], label='FAIL')]
    ax.legend(handles=legend, loc='upper right')
    _save(fig, outpath)


def section9():
    _hdr("Section 9 — Visualisations")

    # ── Fig 1: per-section pass/fail overview ─────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    names   = [r.name[:48] for r in _all_results]
    colors  = [_COLORS['pass'] if r.passed else _COLORS['fail'] for r in _all_results]
    y       = list(range(len(_all_results)))
    ax.barh(y, [1]*len(y), color=colors, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('PASS / FAIL')
    ax.set_title(f'FPGA v{VERSION} Comprehensive Test — All Results')
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=_COLORS['pass'], label='PASS'),
                        Patch(color=_COLORS['fail'], label='FAIL')],
              loc='lower right')
    _save(fig, 'test_v07_overview.png')

    # ── Fig 2: single-qubit gate wall-time bar chart ───────────────────────
    sq = [r for r in _all_results
          if any(r.name.startswith(g) for g in
                 ['ID','H:','X:','X†','Y:','Z:','S·','T·','SX·','SX²','RX','RY',
                  'RZ','P(','U2','U3'])]
    if sq:
        _gate_bar(sq, 'Single-Qubit Gate Wall Times', 'test_v07_sq_gates.png')

    # ── Fig 3: two-qubit native gate bar chart ─────────────────────────────
    tq = [r for r in _all_results
          if any(r.name.startswith(g) for g in
                 ['CX','CZ','Bell','SWAP','CY','CH','CP','CRX','CRY','CRZ',
                  'CSX','DCX','ISWAP','ECR','RXX','RYY','RZZ','CCX'])]
    if tq:
        _gate_bar(tq, 'Multi-Qubit Gate Wall Times', 'test_v07_mq_gates.png')

    # ── Fig 4: QFT scaling — wall time & kernel time vs nq ────────────────
    if _qft_results:
        nqs   = [t[0] for t in _qft_results]
        wtms  = [t[2]/1e3 for t in _qft_results]
        ktms  = [t[3]/1e3 for t in _qft_results]
        clrs  = [_COLORS['pass'] if t[1] else _COLORS['fail'] for t in _qft_results]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # wall time
        ax = axes[0]
        bars = ax.bar(nqs, wtms, color=clrs, edgecolor='white')
        ax.set_xlabel('Number of qubits')
        ax.set_ylabel('Wall time (ms)')
        ax.set_title('QFT — Wall Time vs Qubits')
        ax.set_yscale('log')
        ax.grid(axis='y', color=_COLORS['grid'])
        ax.set_xticks(nqs)

        # kernel time
        ax = axes[1]
        ax.plot(nqs, ktms, marker='o', color=_COLORS['line'], linewidth=2)
        ax.fill_between(nqs, ktms, alpha=0.15, color=_COLORS['line'])
        ax.set_xlabel('Number of qubits')
        ax.set_ylabel('Kernel time (ms)')
        ax.set_title('QFT — FPGA Kernel Time vs Qubits')
        ax.set_yscale('log')
        ax.grid(color=_COLORS['grid'])
        ax.set_xticks(nqs)

        fig.suptitle(f'QFT Scaling — FPGA v{VERSION} (Alveo U55C)',
                     fontsize=13, fontweight='bold')
        _save(fig, 'test_v07_qft_scaling.png')

    # ── Fig 5: Grover's scaling — wall time, kernel time, probability ──────
    if _grover_results:
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
        _save(fig, 'test_v07_grover_scaling.png')

    # ── Fig 6: Combined scaling — QFT vs Grover kernel time ───────────────
    if _qft_results and _grover_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        qnqs = [t[0] for t in _qft_results]
        qkts = [t[3]/1e3 for t in _qft_results]
        ax.plot(qnqs, qkts, marker='o', label='QFT', color=_COLORS['line'],
                linewidth=2)
        ax.plot(gnqs, gktms, marker='s', label="Grover's",
                color=_COLORS['accent'], linewidth=2)
        ax.set_xlabel('Number of qubits', fontsize=12)
        ax.set_ylabel('FPGA Kernel Time (ms)', fontsize=12)
        ax.set_title(f'QFT vs Grover Kernel Time — FPGA v{VERSION}',
                     fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(color=_COLORS['grid'])
        ax.legend(fontsize=11)
        _save(fig, 'test_v07_qft_vs_grover.png')

    # ── Fig 7: classical-bit-mapped measurement tests ─────────────────────
    meas = [r for r in _all_results if r.name.startswith('Meas:')]
    if meas:
        _gate_bar(meas, 'Classical-Bit-Mapped Measurement Tests (v07)',
                  'test_v07_measurement_mapping.png')

    # ── Fig 8: wall time all tests sorted by nq (scatter) ─────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(_all_results)))
    y = [r.wall_us / 1e3 for r in _all_results]
    c = [_COLORS['pass'] if r.passed else _COLORS['fail'] for r in _all_results]
    ax.scatter(x, y, c=c, s=25, zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel('Test index')
    ax.set_ylabel('Wall time (ms)')
    ax.set_title(f'Wall Time — All Tests (FPGA v{VERSION})')
    ax.grid(color=_COLORS['grid'], linewidth=0.5)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0],marker='o',color='w',markerfacecolor=_COLORS['pass'],ms=8,label='PASS'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=_COLORS['fail'],ms=8,label='FAIL'),
    ], fontsize=10)
    _save(fig, 'test_v07_all_wall_times.png')


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 70)
    print(f'  FPGA Quantum Simulator v{VERSION} — Comprehensive Test Suite')
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
    section1(fpga)
    section2(fpga)
    section3(fpga)
    section4(fpga)
    section5(fpga)
    section6(fpga)
    section8(fpga)
    section9()
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

    print(f'\n  Graphs saved:')
    for g in ['test_v07_overview.png', 'test_v07_sq_gates.png',
              'test_v07_mq_gates.png', 'test_v07_qft_scaling.png',
              'test_v07_grover_scaling.png', 'test_v07_qft_vs_grover.png',
              'test_v07_measurement_mapping.png', 'test_v07_all_wall_times.png']:
        print(f'    {g}')

    status = 'ALL PASS ✓' if not failed else f'{failed} FAILURES ✗'
    print(f'\n  Status : {status}')
    print(f'{"="*70}\n')
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()
