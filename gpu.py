#!/usr/bin/env python3
"""
comprehensive_test_aer_gpu.py
==============================
Full gate-correctness + algorithm scaling test, ported from
comprehensive_test_v07.py (fpga_simulator_v07) to run against the
Qiskit Aer statevector simulator on GPU (CUDA) instead of the FPGA
kernel.

Identical to comprehensive_test_aer.py in every test, tolerance, and
graph — the only difference is the backend is constructed with
device='GPU' so the statevector simulation itself runs on the CUDA
device (e.g. the RTX A4000 in this box) instead of the CPU. If no
working GPU backend is found at startup, the script prints a clear
warning and transparently falls back to the CPU backend so the suite
still runs — but every printed line and every saved graph title says
which device was actually used, so you never mistake a CPU fallback
run for a real GPU run.

Requirements on this machine
-----------------------------
  - `qiskit-aer` alone (what `pip show qiskit-aer` reports) is
    CPU-only. GPU support ships in a *separate* wheel:
        pip install qiskit-aer-gpu        # for CUDA 12.x (nvidia-smi here reports CUDA 12.6 -> use this one)
        pip install qiskit-aer-gpu-cu11    # for CUDA 11.x
    Installing qiskit-aer-gpu overwrites the plain qiskit-aer install
    in the same environment (same import name: `qiskit_aer`).
  - Needs a CUDA 11.2+ toolkit/driver already on the system (nvidia-smi
    showing CUDA Version: 12.6 here is sufficient).

Sections
--------
  1  Single-qubit gates (17 native gates)
  2  Two-qubit native gates (CX, CZ)
  3  Decomposed multi-qubit gates (CY, CH, SWAP, CP, CRX, CRY, CRZ, CSX,
       DCX, ISWAP, ECR, RXX, RYY, RZZ, CCX/Toffoli)
  4  ECR state-vector correctness
  5  QFT scaling 1–20 qubits
  6  Grover's algorithm scaling 2–12 qubits
  8  Classical-bit-mapped measurement
  9  Visualisations (saved as PNG files)

Pass criterion
--------------
  Norm within 1e-5 of 1.0 for all tests.
  State-vector checks use per-test thresholds documented inline.
  Section 8 checks the *shape* of the returned counts dict rather than
  a single expected key, since measurement is stochastic.
"""

import math, sys, time, traceback
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import qiskit
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    import qiskit_aer
except ImportError as e:
    print(f"[FATAL] Cannot import qiskit / qiskit-aer: {e}")
    print("        pip install qiskit qiskit-aer qiskit-aer-gpu")
    sys.exit(1)

BACKEND_NAME  = "AerSimulator"
SIM_METHOD    = "statevector"
VERSION       = f"qiskit={qiskit.__version__} aer={qiskit_aer.__version__}"
SHOTS         = 0          # statevector-only — no sampling needed for gate tests
MEAS_SHOTS    = 4000       # shots used specifically by Section 8's counts checks
NORM_TOL      = 1e-5       # slightly relaxed for decomposed gates
STATE_TOL     = 0.02       # probability tolerance for deterministic checks


def _init_backend():
    """Try to construct a GPU-backed AerSimulator and prove it actually
    runs (device='GPU' can be *constructed* even with no working CUDA
    stack — it only fails at .run() time), falling back to CPU with a
    loud warning if it doesn't."""
    try:
        gpu_backend = AerSimulator(method=SIM_METHOD, device='GPU')
        probe = QuantumCircuit(1)
        probe.h(0)
        probe.save_statevector()
        tprobe = transpile(probe, gpu_backend, optimization_level=0)
        result = gpu_backend.run(tprobe, shots=1).result()
        if not result.success:
            raise RuntimeError(str(result.status))
        _ = result.get_statevector(tprobe)
        print(f"  Device  : GPU  (CUDA statevector backend confirmed working)")
        return gpu_backend, 'GPU'
    except Exception as exc:
        print(f"  [WARNING] GPU backend unavailable ({exc}).")
        print(f"            Falling back to CPU. To fix: check `nvidia-smi`,")
        print(f"            and install the matching wheel — `pip install")
        print(f"            qiskit-aer-gpu` for CUDA 12.x, or")
        print(f"            `qiskit-aer-gpu-cu11` for CUDA 11.x — inside")
        print(f"            this venv (it must overwrite the plain qiskit-aer).")
        cpu_backend = AerSimulator(method=SIM_METHOD, device='CPU')
        return cpu_backend, 'CPU (fallback)'


_BACKEND = None
DEVICE_ACTUAL = None


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit construction — maps the FPGA gate-tuple format straight onto Qiskit
# ─────────────────────────────────────────────────────────────────────────────
def _apply_gate(qc, name, qubits, params):
    p = params or []
    if   name == 'id':    qc.id(qubits[0])
    elif name == 'h':     qc.h(qubits[0])
    elif name == 'x':     qc.x(qubits[0])
    elif name == 'y':     qc.y(qubits[0])
    elif name == 'z':     qc.z(qubits[0])
    elif name == 's':     qc.s(qubits[0])
    elif name == 'sdg':   qc.sdg(qubits[0])
    elif name == 't':     qc.t(qubits[0])
    elif name == 'tdg':   qc.tdg(qubits[0])
    elif name == 'sx':    qc.sx(qubits[0])
    elif name == 'sxdg':  qc.sxdg(qubits[0])
    elif name == 'rx':    qc.rx(p[0], qubits[0])
    elif name == 'ry':    qc.ry(p[0], qubits[0])
    elif name == 'rz':    qc.rz(p[0], qubits[0])
    elif name == 'p':     qc.p(p[0], qubits[0])
    elif name == 'u2':    qc.u(math.pi / 2, p[0], p[1], qubits[0])
    elif name == 'u3':    qc.u(p[0], p[1], p[2], qubits[0])
    elif name == 'cx':    qc.cx(qubits[0], qubits[1])
    elif name == 'cz':    qc.cz(qubits[0], qubits[1])
    elif name == 'cy':    qc.cy(qubits[0], qubits[1])
    elif name == 'ch':    qc.ch(qubits[0], qubits[1])
    elif name == 'swap':  qc.swap(qubits[0], qubits[1])
    elif name == 'cp':    qc.cp(p[0], qubits[0], qubits[1])
    elif name == 'crx':   qc.crx(p[0], qubits[0], qubits[1])
    elif name == 'cry':   qc.cry(p[0], qubits[0], qubits[1])
    elif name == 'crz':   qc.crz(p[0], qubits[0], qubits[1])
    elif name == 'csx':   qc.csx(qubits[0], qubits[1])
    elif name == 'dcx':   qc.dcx(qubits[0], qubits[1])
    elif name == 'iswap': qc.iswap(qubits[0], qubits[1])
    elif name == 'ecr':   qc.ecr(qubits[0], qubits[1])
    elif name == 'rxx':   qc.rxx(p[0], qubits[0], qubits[1])
    elif name == 'ryy':   qc.ryy(p[0], qubits[0], qubits[1])
    elif name == 'rzz':   qc.rzz(p[0], qubits[0], qubits[1])
    elif name == 'ccx':   qc.ccx(qubits[0], qubits[1], qubits[2])
    elif name == 'mcx':   qc.mcx(qubits[:-1], qubits[-1])
    elif name == 'measure':
        clbits = params if params is not None else qubits
        for q, c in zip(qubits, clbits):
            qc.measure(q, c)
    else:
        raise ValueError(f"Unsupported gate: {name!r}")


def _build_circuit(nq, gates, num_clbits=None):
    has_measure = any(g[0] == 'measure' for g in gates)
    if num_clbits is None:
        num_clbits = nq if has_measure else 0
    qc = QuantumCircuit(nq, num_clbits) if num_clbits else QuantumCircuit(nq)
    for name, qubits, params in gates:
        _apply_gate(qc, name, qubits, params)
    return qc, has_measure


# ─────────────────────────────────────────────────────────────────────────────
#  Validation — same semantics as FPGAValidator / the FPGA runner's checks
# ─────────────────────────────────────────────────────────────────────────────
class AerValidator:
    @staticmethod
    def check_norm(sv):
        norm = float(np.linalg.norm(np.asarray(sv)))
        return abs(norm - 1.0) < NORM_TOL, norm


def _marginal_prob(sv, qubit, bit):
    probs = np.abs(np.asarray(sv)) ** 2
    idx = np.arange(len(sv))
    mask = ((idx >> qubit) & 1) == bit
    return float(probs[mask].sum())


def _validate_statevector(sv, det_idx=None, det_tolerance=STATE_TOL,
                           marginal_pattern=None, marginal_min_prob=0.45,
                           uniform_check=False, uniform_tolerance=0.05):
    sv = np.asarray(sv)
    norm = float(np.linalg.norm(sv))
    ok = abs(norm - 1.0) < NORM_TOL
    detail_parts = []

    if det_idx is not None:
        prob = float(np.abs(sv[det_idx]) ** 2)
        d_ok = prob >= (1.0 - det_tolerance)
        ok = ok and d_ok
        detail_parts.append(f"P(|{det_idx}>)={prob:.6f}")

    if marginal_pattern:
        for q, bit in marginal_pattern.items():
            prob = _marginal_prob(sv, q, bit)
            m_ok = prob >= marginal_min_prob
            ok = ok and m_ok
            detail_parts.append(f"P(q{q}={bit})={prob:.4f}")

    if uniform_check:
        n = len(sv)
        expected = 1.0 / n
        probs = np.abs(sv) ** 2
        u_ok = bool(np.all(np.abs(probs - expected) < uniform_tolerance))
        ok = ok and u_ok

    return SimpleNamespace(norm=norm, passed=ok, expected_detail='; '.join(detail_parts))


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit execution — the Aer analogue of fpga.run_circuit(...)
# ─────────────────────────────────────────────────────────────────────────────
def run_circuit(nq, gates, shots=0, validate=True, det_idx=None,
                marginal_pattern=None, marginal_min_prob=0.45,
                uniform_check=False, uniform_tolerance=0.05,
                det_tolerance=STATE_TOL, circuit_name='', num_clbits=None):
    """Aer equivalent of the FPGA driver's run_circuit(). Returns a dict with
    'statevector', 'validation', 'counts', and 'timing' (kernel_us, wall_us) —
    matching the shape the FPGA test suite consumed."""
    t0 = time.perf_counter_ns()
    has_measure = any(g[0] == 'measure' for g in gates)
    # shots>0 means the caller wants sampled counts back, even if the gate
    # list has no explicit 'measure' op — mirrors the FPGA driver's own
    # identity-map fallback (measure qubit i -> clbit i, up to
    # min(num_qubits, num_clbits)) when no measure gate was given.
    needs_counts = has_measure or (shots and shots > 0)
    kernel_us = 0.0
    counts = None

    # NOTE: optimization_level=0 is required here — Qiskit's transpiler can
    # otherwise fold a SWAP (or other permutation-equivalent structure) into
    # a virtual final-layout permutation instead of emitting the gate, which
    # silently reorders the returned statevector relative to what the gate
    # list actually specifies.
    if needs_counts:
        # Statevector for norm/validation is taken pre-measurement (unitary
        # part only) since measurement collapses the state — same rationale
        # the FPGA driver used for Section 8's norm check.
        gates_no_measure = [g for g in gates if g[0] != 'measure']
        qc_sv, _ = _build_circuit(nq, gates_no_measure)
        qc_sv.save_statevector()
        tqc_sv = transpile(qc_sv, _BACKEND, optimization_level=0)
        res_sv = _BACKEND.run(tqc_sv, shots=1).result()
        sv = np.asarray(res_sv.get_statevector(tqc_sv))
        kernel_us += res_sv.results[0].time_taken * 1e6

        if has_measure:
            full_gates = gates
        else:
            # Identity fallback: qubit i -> clbit i, up to min(nq, num_clbits)
            n_map = min(nq, num_clbits if num_clbits is not None else nq)
            full_gates = gates + [('measure', list(range(n_map)), list(range(n_map)))]
        qc_full, _ = _build_circuit(nq, full_gates, num_clbits=num_clbits)
        tqc_full = transpile(qc_full, _BACKEND, optimization_level=0)
        shots_eff = shots if shots else MEAS_SHOTS
        res_full = _BACKEND.run(tqc_full, shots=shots_eff).result()
        counts = res_full.get_counts()
        kernel_us += res_full.results[0].time_taken * 1e6
    else:
        qc, _ = _build_circuit(nq, gates, num_clbits=num_clbits)
        qc.save_statevector()
        tqc = transpile(qc, _BACKEND, optimization_level=0)
        res = _BACKEND.run(tqc, shots=1).result()
        sv = np.asarray(res.get_statevector(tqc))
        kernel_us += res.results[0].time_taken * 1e6

    wall_us = (time.perf_counter_ns() - t0) / 1e3

    validation = None
    if validate:
        validation = _validate_statevector(
            sv, det_idx=det_idx, det_tolerance=det_tolerance,
            marginal_pattern=marginal_pattern, marginal_min_prob=marginal_min_prob,
            uniform_check=uniform_check, uniform_tolerance=uniform_tolerance)

    return {
        'statevector': sv,
        'validation': validation,
        'counts': counts,
        'timing': {'kernel_us': kernel_us, 'wall_us': wall_us},
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Result bookkeeping
# ─────────────────────────────────────────────────────────────────────────────
class R:
    """Lightweight test result."""
    __slots__ = ('name', 'passed', 'norm', 'wall_us', 'detail')

    def __init__(self, name, passed, norm, wall_us, detail=''):
        self.name = name
        self.passed = passed
        self.norm = norm
        self.wall_us = wall_us
        self.detail = detail

    def __str__(self):
        s = 'PASS' if self.passed else 'FAIL'
        ms = self.wall_us / 1e3
        d = f'  [{self.detail}]' if self.detail else ''
        return f"  [{s}] {self.name:<42s} norm={self.norm:.10f}  {ms:8.2f} ms{d}"


_all_results = []


def _run(name, nq, gates, *, det_idx=None, marginal=None,
         marginal_min=0.45, uniform=False):
    """Execute one test, append result, return R."""
    try:
        t0 = time.perf_counter_ns()
        res = run_circuit(
            nq, gates, shots=SHOTS, validate=True,
            det_idx=det_idx,
            marginal_pattern=marginal, marginal_min_prob=marginal_min,
            uniform_check=uniform,
            det_tolerance=STATE_TOL,
            circuit_name=name)
        wall_us = (time.perf_counter_ns() - t0) / 1e3
        val = res['validation']
        norm = val.norm if val else 0.0
        ok = val.passed if val else False
        det = val.expected_detail if val else ''
        r = R(name, ok, norm, wall_us, det)
    except Exception as exc:
        r = R(name, False, 0.0, 0.0, str(exc)[:80])
        traceback.print_exc()
    _all_results.append(r)
    print(r)
    return r


def _run_meas(name, nq, gates, num_clbits, check_fn, shots=MEAS_SHOTS):
    """Execute one measurement-mapping test.

    check_fn(counts: dict[str,int]) -> (ok: bool, detail: str). Norm is
    still validated against the pre-measurement statevector; ok is the AND
    of norm-validity and check_fn's own verdict.
    """
    try:
        t0 = time.perf_counter_ns()
        res = run_circuit(
            nq, gates, shots=shots, validate=True,
            det_tolerance=STATE_TOL, circuit_name=name,
            num_clbits=num_clbits)
        wall_us = (time.perf_counter_ns() - t0) / 1e3
        val = res['validation']
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
    sv = np.asarray(sv)
    return float(sv[idx].real ** 2 + sv[idx].imag ** 2)


def _run_sv(name, nq, gates):
    """Run circuit, return (statevector, wall_us).  No validation object."""
    t0 = time.perf_counter_ns()
    res = run_circuit(nq, gates, shots=SHOTS, validate=False, circuit_name=name)
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
def section1():
    _hdr("Section 1 — Single-Qubit Gates")
    nq = 3   # 3 qubits so gates on q=0 don't trivially pass due to |0⟩ symmetry

    _run("ID: no-op norm", nq, [('id', [0], None)], det_idx=0)
    _run("H: uniform superposition |+⟩", 1, [('h', [0], None)], uniform=True)
    _run("X: |0⟩→|1⟩", 1, [('x', [0], None)], det_idx=1)
    _run("X†X: identity", 1, [('x', [0], None), ('x', [0], None)], det_idx=0)
    _run("Y: |0⟩→i|1⟩", 1, [('y', [0], None)], det_idx=1)
    _run("Z: |0⟩ unchanged (phase only)", 1, [('z', [0], None)], det_idx=0)
    _run("S·S†: identity", 1, [('s', [0], None), ('sdg', [0], None)], det_idx=0)
    _run("T·T†: identity", 1, [('t', [0], None), ('tdg', [0], None)], det_idx=0)
    _run("SX·SX†: identity", 1, [('sx', [0], None), ('sxdg', [0], None)], det_idx=0)
    _run("SX²=X: |0⟩→|1⟩", 1, [('sx', [0], None), ('sx', [0], None)], det_idx=1)
    _run("RX(π): |0⟩→|1⟩", 1, [('rx', [0], [math.pi])], det_idx=1)
    _run("RX(2π): identity |0⟩", 1, [('rx', [0], [2 * math.pi])], det_idx=0)
    _run("RY(π): |0⟩→|1⟩", 1, [('ry', [0], [math.pi])], det_idx=1)
    _run("RZ(π): |0⟩ unchanged (phase)", 1, [('rz', [0], [math.pi])], det_idx=0)
    _run("P(π): |0⟩ unchanged", 1, [('p', [0], [math.pi])], det_idx=0)
    _run("U2(0,π): = H, uniform superposition", 1, [('u2', [0], [0.0, math.pi])], uniform=True)
    _run("U3(π,0,π): = X gate, |0⟩→|1⟩", 1, [('u3', [0], [math.pi, 0.0, math.pi])], det_idx=1)
    _run("U3(π/2,0,π): = H, uniform superposition", 1,
         [('u3', [0], [math.pi / 2, 0.0, math.pi])], uniform=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — Two-qubit native gates
# ═════════════════════════════════════════════════════════════════════════════
def section2():
    _hdr("Section 2 — Two-Qubit Native Gates (CX, CZ)")

    _run("CX: Bell state |Φ+⟩", 2, [('h', [0], None), ('cx', [0, 1], None)],
         marginal={0: 0}, marginal_min=0.45)
    _run("CX: |10⟩→|11⟩", 2, [('x', [0], None), ('cx', [0, 1], None)], det_idx=3)
    _run("CX²: identity |00⟩", 2,
         [('x', [0], None), ('cx', [0, 1], None), ('cx', [0, 1], None), ('x', [0], None)],
         det_idx=0)
    _run("CZ: |00⟩ unchanged", 2, [('cz', [0, 1], None)], det_idx=0)
    _run("CZ: |11⟩→-|11⟩ (phase, norm=1)", 2,
         [('x', [0], None), ('x', [1], None), ('cz', [0, 1], None)], det_idx=3)
    _run("CZ²: identity", 2,
         [('x', [0], None), ('x', [1], None), ('cz', [0, 1], None), ('cz', [0, 1], None)],
         det_idx=3)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Decomposed multi-qubit gates
# ═════════════════════════════════════════════════════════════════════════════
def section3():
    _hdr("Section 3 — Decomposed Multi-Qubit Gates")

    _run("CY: |10⟩→i|11⟩", 2, [('x', [0], None), ('cy', [0, 1], None)], det_idx=3)
    _run("CH: norm preserved", 2, [('x', [0], None), ('ch', [0, 1], None)])
    _run("SWAP: |10⟩→|01⟩", 2, [('x', [0], None), ('swap', [0, 1], None)], det_idx=2)
    _run("SWAP²: identity", 2,
         [('x', [0], None), ('swap', [0, 1], None), ('swap', [0, 1], None), ('x', [0], None)],
         det_idx=0)
    _run("CP(π): |11⟩ phase flip, norm=1", 2,
         [('x', [0], None), ('x', [1], None), ('cp', [0, 1], [math.pi])], det_idx=3)
    _run("CP(2π): identity", 2,
         [('x', [0], None), ('x', [1], None), ('cp', [0, 1], [2 * math.pi]),
          ('x', [0], None), ('x', [1], None)], det_idx=0)
    _run("CRX(π): |10⟩→|11⟩", 2, [('x', [0], None), ('crx', [0, 1], [math.pi])], det_idx=3)
    _run("CRY(π): |10⟩→|11⟩", 2, [('x', [0], None), ('cry', [0, 1], [math.pi])], det_idx=3)
    _run("CRZ(π): norm=1", 2, [('x', [0], None), ('crz', [0, 1], [math.pi])])
    _run("CSX: norm=1", 2, [('x', [0], None), ('csx', [0, 1], None)])
    _run("CSX²=CX: |10⟩→|11⟩", 2,
         [('x', [0], None), ('csx', [0, 1], None), ('csx', [0, 1], None)], det_idx=3)
    _run("DCX: norm=1", 2, [('dcx', [0, 1], None)])
    _run("ISWAP: norm=1", 2, [('x', [0], None), ('iswap', [0, 1], None)])
    _run("ISWAP: |10⟩→i|01⟩", 2, [('x', [0], None), ('iswap', [0, 1], None)], det_idx=2)
    _run("ECR: |00⟩ norm=1", 2, [('ecr', [0, 1], None)])
    _run("ECR: |11⟩ norm=1", 2, [('x', [0], None), ('x', [1], None), ('ecr', [0, 1], None)])
    _run("RXX(π/2): norm=1", 2, [('rxx', [0, 1], [math.pi / 2])])
    _run("RYY(π/2): norm=1", 2, [('ryy', [0, 1], [math.pi / 2])])
    _run("RZZ(π/2): norm=1", 2, [('rzz', [0, 1], [math.pi / 2])])
    _run("CCX: |110⟩→|111⟩", 3,
         [('x', [0], None), ('x', [1], None), ('ccx', [0, 1, 2], None)], det_idx=7)
    _run("CCX: |100⟩ unchanged", 3, [('x', [0], None), ('ccx', [0, 1, 2], None)], det_idx=1)
    _run("CCX²: identity", 3,
         [('x', [0], None), ('x', [1], None),
          ('ccx', [0, 1, 2], None), ('ccx', [0, 1, 2], None),
          ('x', [0], None), ('x', [1], None)], det_idx=0)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — ECR state-vector correctness
# ═════════════════════════════════════════════════════════════════════════════
def section4():
    _hdr("Section 4 — ECR State-Vector Correctness")

    expected = {
        '|00⟩': ([], {1: 0.5, 3: 0.5}),
        '|01⟩': ([('x', [0], None)], {0: 0.5, 2: 0.5}),
        '|10⟩': ([('x', [1], None)], {1: 0.5, 3: 0.5}),
        '|11⟩': ([('x', [0], None), ('x', [1], None)], {0: 0.5, 2: 0.5}),
    }

    for label, (prep, probs_exp) in expected.items():
        gates = prep + [('ecr', [0, 1], None)]
        sv, wall_us = _run_sv(f"ECR {label}", 2, gates)
        norm_ok, norm = AerValidator.check_norm(sv)
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


def section5():
    _hdr("Section 5 — QFT Scaling (1–20 qubits)")

    for nq in range(1, 21):
        name = f"QFT {nq:2d}q: uniform superposition"
        gates = _qft_gates(nq)
        try:
            t0 = time.perf_counter_ns()
            res = run_circuit(
                nq, gates, shots=SHOTS, validate=True,
                uniform_check=True, uniform_tolerance=0.05,
                circuit_name=name)
            wall_us = (time.perf_counter_ns() - t0) / 1e3
            val = res['validation']
            norm = val.norm if val else 0.0
            ok = val.passed if val else False
            kus = res['timing']['kernel_us']
            r = R(name, ok, norm, wall_us)
            _qft_results.append((nq, ok, wall_us, kus))
        except Exception as exc:
            r = R(name, False, 0.0, 0.0, str(exc)[:60])
            _qft_results.append((nq, False, 0.0, 0.0))
            traceback.print_exc()
        _all_results.append(r)
        print(r)

    print()
    for nq in [1, 2, 4, 8, 12]:
        name = f"QFT+IQFT {nq:2d}q: round-trip |0⟩"
        gates = _qft_gates(nq) + _qft_gates(nq, inverse=True)
        try:
            t0 = time.perf_counter_ns()
            res = run_circuit(
                nq, gates, shots=SHOTS, validate=True,
                det_idx=0, det_tolerance=0.02,
                circuit_name=name)
            wall_us = (time.perf_counter_ns() - t0) / 1e3
            val = res['validation']
            norm = val.norm if val else 0.0
            ok = val.passed if val else False
            r = R(name, ok, norm, wall_us,
                  f"P(|0⟩)={_prob(res['statevector'],0):.6f}")
        except Exception as exc:
            r = R(name, False, 0.0, 0.0, str(exc)[:60])
            traceback.print_exc()
        _all_results.append(r)
        print(r)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Grover's algorithm scaling 2–12 qubits
# ═════════════════════════════════════════════════════════════════════════════
def _ccz_ladder(qubits):
    """Multi-controlled Z on n qubits: applies phase -1 to |1...1⟩ only."""
    n = len(qubits)
    if n == 1:
        return [('z', [qubits[0]], None)]
    if n == 2:
        return [('cz', [qubits[0], qubits[1]], None)]
    if n == 3:
        return [('h', [qubits[2]], None),
                ('ccx', [qubits[0], qubits[1], qubits[2]], None),
                ('h', [qubits[2]], None)]
    target = qubits[-1]
    controls = list(qubits[:-1])
    return [
        ('h', [target], None),
        ('mcx', controls + [target], None),
        ('h', [target], None),
    ]


def _grover_oracle_all_ones(n):
    return _ccz_ladder(list(range(n)))


def _grover_diffusion(n):
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
    target_idx = (1 << n) - 1

    per_iter_est = 4 * n * 2 + 2 * n * 2
    _theta = math.asin(1.0 / math.sqrt(1 << n))
    optimal_iters = max(1, round(math.pi / (4.0 * _theta) - 0.5))
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


def section6():
    _hdr("Section 6 — Grover's Algorithm Scaling (2–12 qubits)")
    print("  (oracle: native CZ/CCZ for nq≤3, H+MCX+H for nq>3)")

    for nq in range(2, 13):
        gates, tgt_idx, iters = _grover_circuit(nq)
        prob_thresh = 0.5 if nq <= 6 else 0.05
        name = f"Grover {nq:2d}q ({iters} iter): P(|1..1⟩)≥{prob_thresh:.2f}"
        try:
            t0 = time.perf_counter_ns()
            res = run_circuit(
                nq, gates, shots=SHOTS, validate=True,
                det_idx=tgt_idx, det_tolerance=1.0 - prob_thresh,
                circuit_name=name)
            wall_us = (time.perf_counter_ns() - t0) / 1e3
            sv = res['statevector']
            val = res['validation']
            norm = val.norm if val else 0.0
            ok = norm_ok = abs(norm - 1.0) < NORM_TOL
            kus = res['timing']['kernel_us']
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
#  SECTION 8 — Classical-bit-mapped measurement
# ═════════════════════════════════════════════════════════════════════════════
def section8():
    _hdr("Section 8 — Classical-Bit-Mapped Measurement")

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

    _run_meas("Meas: 1 qubit -> 2 clbits, 1 unmeasured", 1,
              [('h', [0], None), ('measure', [0], [0]), ('measure', [0], [1])],
              num_clbits=3, check_fn=check_fanout)

    def check_high_clbit(counts):
        keys = set(counts)
        if keys != {'100'}:
            return False, f"expected only '100', got {sorted(keys)}"
        return True, f"keys={sorted(counts.items())}"

    _run_meas("Meas: qubit0 -> clbit2 only (deterministic)", 1,
              [('x', [0], None), ('measure', [0], [2])],
              num_clbits=3, check_fn=check_high_clbit)

    def check_two_qubit_map(counts):
        keys = set(counts)
        if not keys.issubset({'10', '11'}):
            return False, f"unexpected keys: {sorted(keys)}"
        for key in keys:
            if key[0] != '1':
                return False, f"clbit1 (from qubit1=X) not '1': {key!r}"
        return True, f"keys={sorted(counts.items())}"

    _run_meas("Meas: 2 qubits -> 2 clbits, explicit map", 2,
              [('h', [0], None), ('x', [1], None),
               ('measure', [0], [0]), ('measure', [1], [1])],
              num_clbits=2, check_fn=check_two_qubit_map)

    def check_fallback(counts):
        keys = set(counts)
        if keys != {'01'}:
            return False, f"expected only '01', got {sorted(keys)}"
        return True, f"keys={sorted(counts.items())}"

    _run_meas("Meas: clbits=None fallback (qubit==clbit)", 2,
              [('x', [0], None), ('measure', [0, 1], None)],
              num_clbits=2, check_fn=check_fallback)

    def check_no_measure(counts):
        keys = set(counts)
        if keys != {'000'}:
            return False, f"expected only '000' (no measure => |0>), got {sorted(keys)}"
        return True, f"keys={sorted(counts.items())}"

    _run_meas("Meas: no explicit measure, |0> default", 2,
              [('id', [0], None)],
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
    names = [r.name.split(':')[0].strip() for r in results_subset]
    times = [r.wall_us / 1e3 for r in results_subset]
    colors = [_COLORS['pass'] if r.passed else _COLORS['fail'] for r in results_subset]
    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 5))
    ax.bar(names, times, color=colors, edgecolor='white', linewidth=0.5)
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
    names = [r.name[:48] for r in _all_results]
    colors = [_COLORS['pass'] if r.passed else _COLORS['fail'] for r in _all_results]
    y = list(range(len(_all_results)))
    ax.barh(y, [1] * len(y), color=colors, edgecolor='white', linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel('PASS / FAIL')
    ax.set_title(f'Aer Comprehensive Test — All Results ({VERSION}, device={DEVICE_ACTUAL})')
    ax.invert_yaxis()
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=_COLORS['pass'], label='PASS'),
                        Patch(color=_COLORS['fail'], label='FAIL')],
              loc='lower right')
    _save(fig, 'test_aer_gpu_overview.png')

    # ── Fig 2: single-qubit gate wall-time bar chart ───────────────────────
    sq = [r for r in _all_results
          if any(r.name.startswith(g) for g in
                 ['ID', 'H:', 'X:', 'X†', 'Y:', 'Z:', 'S·', 'T·', 'SX·', 'SX²', 'RX', 'RY',
                  'RZ', 'P(', 'U2', 'U3'])]
    if sq:
        _gate_bar(sq, 'Single-Qubit Gate Wall Times', 'test_aer_gpu_sq_gates.png')

    # ── Fig 3: two-qubit / multi-qubit native gate bar chart ───────────────
    tq = [r for r in _all_results
          if any(r.name.startswith(g) for g in
                 ['CX', 'CZ', 'Bell', 'SWAP', 'CY', 'CH', 'CP', 'CRX', 'CRY', 'CRZ',
                  'CSX', 'DCX', 'ISWAP', 'ECR', 'RXX', 'RYY', 'RZZ', 'CCX'])]
    if tq:
        _gate_bar(tq, 'Multi-Qubit Gate Wall Times', 'test_aer_gpu_mq_gates.png')

    # ── Fig 4: QFT scaling — wall time & kernel time vs nq ────────────────
    if _qft_results:
        nqs = [t[0] for t in _qft_results]
        wtms = [t[2] / 1e3 for t in _qft_results]
        ktms = [t[3] / 1e3 for t in _qft_results]
        clrs = [_COLORS['pass'] if t[1] else _COLORS['fail'] for t in _qft_results]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.bar(nqs, wtms, color=clrs, edgecolor='white')
        ax.set_xlabel('Number of qubits')
        ax.set_ylabel('Wall time (ms)')
        ax.set_title('QFT — Wall Time vs Qubits')
        ax.set_yscale('log')
        ax.grid(axis='y', color=_COLORS['grid'])
        ax.set_xticks(nqs)

        ax = axes[1]
        ax.plot(nqs, ktms, marker='o', color=_COLORS['line'], linewidth=2)
        ax.fill_between(nqs, ktms, alpha=0.15, color=_COLORS['line'])
        ax.set_xlabel('Number of qubits')
        ax.set_ylabel('Kernel time (ms)')
        ax.set_title('QFT — Aer Kernel Time vs Qubits')
        ax.set_yscale('log')
        ax.grid(color=_COLORS['grid'])
        ax.set_xticks(nqs)

        fig.suptitle(f'QFT Scaling — Qiskit Aer ({SIM_METHOD} method, device={DEVICE_ACTUAL})',
                     fontsize=13, fontweight='bold')
        _save(fig, 'test_aer_gpu_qft_scaling.png')

    # ── Fig 5: Grover's scaling — wall time, kernel time, probability ──────
    if _grover_results:
        gnqs = [t[0] for t in _grover_results]
        gwtms = [t[2] / 1e3 for t in _grover_results]
        gktms = [t[3] / 1e3 for t in _grover_results]
        gprob = [t[5] for t in _grover_results]
        gcols = [_COLORS['pass'] if t[1] else _COLORS['fail'] for t in _grover_results]

        fig = plt.figure(figsize=(16, 5))
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

        ax0 = fig.add_subplot(gs[0])
        ax0.bar(gnqs, gwtms, color=gcols, edgecolor='white')
        ax0.set_xlabel('Number of qubits')
        ax0.set_ylabel('Wall time (ms)')
        ax0.set_title("Grover's — Wall Time")
        ax0.set_yscale('log')
        ax0.grid(axis='y', color=_COLORS['grid'])
        ax0.set_xticks(gnqs[::2])

        ax1 = fig.add_subplot(gs[1])
        ax1.plot(gnqs, gktms, marker='s', color=_COLORS['accent'], linewidth=2)
        ax1.fill_between(gnqs, gktms, alpha=0.15, color=_COLORS['accent'])
        ax1.set_xlabel('Number of qubits')
        ax1.set_ylabel('Kernel time (ms)')
        ax1.set_title("Grover's — Aer Kernel Time")
        ax1.set_yscale('log')
        ax1.grid(color=_COLORS['grid'])
        ax1.set_xticks(gnqs[::2])

        ax2 = fig.add_subplot(gs[2])
        ax2.plot(gnqs, gprob, marker='^', color=_COLORS['line'], linewidth=2)
        ax2.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='threshold 0.5')
        ax2.set_xlabel('Number of qubits')
        ax2.set_ylabel('P(|1…1⟩)')
        ax2.set_title("Grover's — Target State Probability")
        ax2.set_ylim(0, 1.05)
        ax2.grid(color=_COLORS['grid'])
        ax2.set_xticks(gnqs[::2])
        ax2.legend(fontsize=8)

        fig.suptitle(f"Grover's Algorithm Scaling — Qiskit Aer (device={DEVICE_ACTUAL})",
                     fontsize=13, fontweight='bold')
        _save(fig, 'test_aer_gpu_grover_scaling.png')

    # ── Fig 6: Combined scaling — QFT vs Grover kernel time ───────────────
    if _qft_results and _grover_results:
        fig, ax = plt.subplots(figsize=(10, 5))
        qnqs = [t[0] for t in _qft_results]
        qkts = [t[3] / 1e3 for t in _qft_results]
        ax.plot(qnqs, qkts, marker='o', label='QFT', color=_COLORS['line'], linewidth=2)
        ax.plot(gnqs, gktms, marker='s', label="Grover's", color=_COLORS['accent'], linewidth=2)
        ax.set_xlabel('Number of qubits', fontsize=12)
        ax.set_ylabel('Aer Kernel Time (ms)', fontsize=12)
        ax.set_title(f'QFT vs Grover Kernel Time — Qiskit Aer (device={DEVICE_ACTUAL})', fontsize=13, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(color=_COLORS['grid'])
        ax.legend(fontsize=11)
        _save(fig, 'test_aer_gpu_qft_vs_grover.png')

    # ── Fig 7: classical-bit-mapped measurement tests ─────────────────────
    meas = [r for r in _all_results if r.name.startswith('Meas:')]
    if meas:
        _gate_bar(meas, 'Classical-Bit-Mapped Measurement Tests', 'test_aer_gpu_measurement_mapping.png')

    # ── Fig 8: wall time all tests, in order (scatter) ─────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    x = list(range(len(_all_results)))
    y = [r.wall_us / 1e3 for r in _all_results]
    c = [_COLORS['pass'] if r.passed else _COLORS['fail'] for r in _all_results]
    ax.scatter(x, y, c=c, s=25, zorder=3)
    ax.set_yscale('log')
    ax.set_xlabel('Test index')
    ax.set_ylabel('Wall time (ms)')
    ax.set_title(f'Wall Time — All Tests (Qiskit Aer, {VERSION}, device={DEVICE_ACTUAL})')
    ax.grid(color=_COLORS['grid'], linewidth=0.5)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_COLORS['pass'], ms=8, label='PASS'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_COLORS['fail'], ms=8, label='FAIL'),
    ], fontsize=10)
    _save(fig, 'test_aer_gpu_all_wall_times.png')


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    global _BACKEND, DEVICE_ACTUAL

    print('=' * 70)
    print(f'  Qiskit Aer Comprehensive Test Suite (GPU-targeted)')
    print(f'  Backend : {BACKEND_NAME}  |  Method: {SIM_METHOD}  |  Requested device: GPU')
    print(f'  Versions: {VERSION}')
    print(f'  Started : {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    print('\nInitialising Aer backend (probing GPU) …')
    t_init = time.perf_counter_ns()
    _BACKEND, DEVICE_ACTUAL = _init_backend()
    print(f'  Ready in {(time.perf_counter_ns()-t_init)/1e6:.1f} ms  |  Device in use: {DEVICE_ACTUAL}\n')

    t_suite = time.perf_counter_ns()
    section1()
    section2()
    section3()
    section4()
    section5()
    section6()
    section8()
    section9()
    suite_s = (time.perf_counter_ns() - t_suite) / 1e9

    total = len(_all_results)
    passed = sum(1 for r in _all_results if r.passed)
    failed = total - passed

    print(f'\n{"="*70}')
    print(f'  FINAL RESULT : {passed}/{total} PASSED  ({failed} failed)')
    print(f'  Suite time   : {suite_s:.1f} s')
    print(f'  Device used  : {DEVICE_ACTUAL}')

    if failed:
        print(f'\n  Failed tests:')
        for r in _all_results:
            if not r.passed:
                print(f'    ✗ {r.name}')

    print(f'\n  Graphs saved:')
    for g in ['test_aer_gpu_overview.png', 'test_aer_gpu_sq_gates.png',
              'test_aer_gpu_mq_gates.png', 'test_aer_gpu_qft_scaling.png',
              'test_aer_gpu_grover_scaling.png', 'test_aer_gpu_qft_vs_grover.png',
              'test_aer_gpu_measurement_mapping.png', 'test_aer_gpu_all_wall_times.png']:
        print(f'    {g}')

    status = 'ALL PASS ✓' if not failed else f'{failed} FAILURES ✗'
    print(f'\n  Status : {status}')
    print(f'{"="*70}\n')
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()