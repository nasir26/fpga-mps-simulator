"""
Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
Target platform: Xilinx Alveo U55C × 4, Vitis 2023.2, XRT 2.x, pyxrt, Python 3.8
"""

from __future__ import annotations

# ── spawn MUST be set before any other multiprocessing usage ─────────────────
import multiprocessing as _mp
try:
    _mp.set_start_method('spawn', force=False)
except RuntimeError:
    pass

import atexit
import logging
import math
import os
import signal
import struct
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

from multiprocessing import Event, Process, Queue

# ── XRT import ────────────────────────────────────────────────────────────────
try:
    import pyxrt as xrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False

# ── Qiskit (optional) ────────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# ── Hybrid CPU-side MPS primitives (gate matrices, SVD/QR, sampling) ─────────
# NOTE: this is NOT a CPU fallback simulator. mps_core.py contains only the
# tensor-chain container, gate-matrix construction, and the CPU-side
# SVD/QR/sampling steps that are part of the hybrid design by definition.
# Gate/tensor contraction itself is only ever performed by the FPGA kernel.
from mps_core import (
    MPSState,
    GATE_MATRICES, _gate_1q, _gate_2q,
    _SINGLE_QUBIT_GATES, _TWO_QUBIT_GATES,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Logging
# ─────────────────────────────────────────────────────────────────────────────
_lvl = (logging.DEBUG
        if os.environ.get('FPGA_MPS_LOG', '').upper() == 'DEBUG'
        else logging.WARNING)
logging.basicConfig(level=_lvl,
                    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger('fpga_mps')

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
NUM_CARDS      = 1
NUM_HBM_BANKS  = 8          # 8 tensor + 1 gate bundle = 9 AXI masters
KERNEL_NAME    = 'fpga_mps_simulator'
DEFAULT_XCLBIN = 'fpga_mps_simulator.xclbin'

INSTR_WORDS    = 12         # header words per gate instruction
GATE_MAT_WORDS = 32         # 16 complex128 as float32 pairs → 32 int32
WORDS_PER_GATE = INSTR_WORDS + GATE_MAT_WORDS   # 44 int32 per gate
MAX_GATES_BRAM = 512        # must match fpga_mps_simulator.cpp MAX_GATES_BRAM

# contract_and_apply_2q's theta_re/theta_im URAM buffers are sized
# CHI_MAX_SVD x 4 x CHI_MAX_SVD (see fpga_mps_simulator.cpp) and every loop
# over chi_l/chi_m/chi_r is clamped to this bound with no error signalled to
# the host on overflow -- indices at or beyond it are silently dropped from
# the contraction, so chi_max here must never exceed it.
CHI_MAX_SVD_HW = 64         # must match fpga_mps_simulator.cpp CHI_MAX_SVD

# Opcodes (must match fpga_mps_simulator.cpp)
_OP1Q: Dict[str, int] = {
    'id': 0, 'h': 1, 'x': 2, 'y': 3, 'z': 4,
    's': 5, 'sdg': 6, 't': 7, 'tdg': 8, 'sx': 9, 'sxdg': 10,
    'rx': 11, 'ry': 12, 'rz': 13, 'p': 14, 'phase': 14, 'u1': 14,
    'u2': 15, 'u3': 16, 'u': 16,
}
_OP2Q: Dict[str, int] = {
    'cx': 17, 'cnot': 17, 'cy': 18, 'cz': 19, 'ch': 20,
    'cp': 21, 'cphase': 21, 'crx': 22, 'cry': 23, 'crz': 24,
    'csx': 25, 'swap': 26, 'iswap': 27, 'dcx': 28,
    'ecr': 29, 'rxx': 30, 'ryy': 31, 'rzz': 32,
}
OP_2Q_CONTRACT = 100  # kernel: contract+gate only; host does SVD
OP_2Q_FULL     = 102  # kernel: contract+gate+JacobiSVD+split (chi ≤ 64)

CHI_FULL_SVD   = 0    # SVD always runs on host CPU (Jacobi SVD removed from FPGA)


# ─────────────────────────────────────────────────────────────────────────────
#  Float32-in-int32 packing (union trick, matches C++ extract_f32)
# ─────────────────────────────────────────────────────────────────────────────

def _f32_bits(f: float) -> int:
    """Pack a float as its IEEE 754 float32 bits in a Python int."""
    return struct.unpack('<I', struct.pack('<f', float(np.float32(f))))[0]


# ─────────────────────────────────────────────────────────────────────────────
#  HBM bump allocator
# ─────────────────────────────────────────────────────────────────────────────

class _HBMAlloc:
    """
    Bump allocator for complex-double slots in 8-bank HBM.
    Offset in units of complex128 amplitudes (= 2×float64 = 16 bytes).
    """

    def __init__(self):
        self._cursor = 0

    def alloc(self, n_amps: int) -> int:
        off = self._cursor
        # Round up to next multiple of NUM_HBM_BANKS so every site's g_offset
        # is bank-aligned.  _hbm_write/_hbm_read assume g_offset % 8 == 0:
        # local_index i maps to bank i%8, which only holds when g_offset%8 == 0.
        n_padded = ((n_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS) * NUM_HBM_BANKS
        self._cursor += n_padded
        return off

    def alloc_site(self, chi_l: int, chi_r: int) -> int:
        return self.alloc(chi_l * 2 * chi_r)   # σ ∈ {0,1}

    @property
    def total(self) -> int:
        return self._cursor


# ─────────────────────────────────────────────────────────────────────────────
#  HBM tensor I/O via pyxrt BO objects
# ─────────────────────────────────────────────────────────────────────────────

def _hbm_write(banks: list, tensor: np.ndarray,
               g_offset: int, n_amps: int, xrt_mod):
    """Write n_amps complex128 values to HBM (8-bank interleaved)."""
    c = tensor.ravel()[:n_amps].astype(np.complex128)
    amps_per_bank = (n_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS

    for bid in range(NUM_HBM_BANKS):
        idxs = np.arange(bid, n_amps, NUM_HBM_BANKS)
        if len(idxs) == 0:
            continue
        buf = np.zeros(amps_per_bank * 2, dtype=np.float64)
        v = len(idxs)
        buf[:v * 2:2] = c[idxs].real
        buf[1:v * 2:2] = c[idxs].imag
        byte_off = (g_offset // NUM_HBM_BANKS) * 2 * 8
        n_bytes  = v * 2 * 8
        banks[bid].write(buf.tobytes(), byte_off)
        banks[bid].sync(
            xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
            n_bytes, byte_off
        )


def _hbm_read(banks: list, g_offset: int,
              n_amps: int, shape: tuple, xrt_mod) -> np.ndarray:
    """Read n_amps complex128 values from HBM (8-bank interleaved)."""
    amps_per_bank = (n_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
    result = np.zeros(n_amps, dtype=np.complex128)
    # g_offset is always a multiple of 8 (enforced by _HBMAlloc.alloc padding),
    # so slot_start = g_offset // 8 is the starting slot index in every bank.
    byte_off  = (g_offset // NUM_HBM_BANKS) * 2 * 8  # = slot_start * 16
    f64_start = byte_off // 8                          # index into float64 view

    for bid in range(NUM_HBM_BANKS):
        idxs = np.arange(bid, n_amps, NUM_HBM_BANKS)
        if len(idxs) == 0:
            continue
        n_bytes = len(idxs) * 2 * 8
        banks[bid].sync(
            xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
            n_bytes, byte_off
        )
        # Read starting at byte_off (slot_start), not from byte 0.
        raw_all = np.frombuffer(banks[bid].map(), dtype=np.float64)
        raw = raw_all[f64_start : f64_start + amps_per_bank * 2].copy()
        v = min(len(idxs), amps_per_bank)
        result[idxs[:v]] = raw[:v * 2:2] + 1j * raw[1:v * 2:2]

    return result.reshape(shape)


# ─────────────────────────────────────────────────────────────────────────────
#  Gate instruction encoder
# ─────────────────────────────────────────────────────────────────────────────

def _enc_header(opcode: int, si: int, sj: int,
                chi_l: int, chi_m: int, chi_r: int, chi_new: int,
                params: Optional[list],
                off_i: int, off_j: int) -> np.ndarray:
    h = np.zeros(INSTR_WORDS, dtype=np.int32)
    h[0] = opcode
    h[1] = si
    h[2] = sj
    h[3] = chi_l
    h[4] = chi_m
    h[5] = chi_r
    h[6] = chi_new
    for k, p in enumerate((params or [])[:3]):
        h[7 + k] = _f32_bits(p)
    h[10] = off_i
    h[11] = off_j
    return h


def _build_1q_instr(opcode: int, site: int,
                    chi_l: int, chi_r: int,
                    params: Optional[list],
                    off_i: int,
                    gate_mat: np.ndarray) -> np.ndarray:
    instr = np.zeros(WORDS_PER_GATE, dtype=np.int32)
    instr[:INSTR_WORDS] = _enc_header(
        opcode, site, -1, chi_l, 0, chi_r, 0, params, off_i, -1
    )
    # 2×2 complex128 → 8 float32 pairs
    g2 = gate_mat.ravel().astype(np.complex128)
    for k in range(4):
        instr[INSTR_WORDS + k * 2]     = _f32_bits(g2[k].real)
        instr[INSTR_WORDS + k * 2 + 1] = _f32_bits(g2[k].imag)
    # The compiled xclbin reads p0,p1,p2 from gbuf[base+INSTR_WORDS+{0,1,2}]
    # (words 12,13,14) not from the header words 7,8,9.  Override those slots.
    ps = list((params or [])[:3]) + [0.0, 0.0, 0.0]
    for k in range(3):
        instr[INSTR_WORDS + k] = _f32_bits(ps[k])
    return instr


def _build_2q_instr(opcode: int, si: int, sj: int,
                    chi_l: int, chi_m: int, chi_r: int, chi_new: int,
                    params: Optional[list],
                    off_i: int, off_j: int,
                    gate_mat: np.ndarray) -> np.ndarray:
    instr = np.zeros(WORDS_PER_GATE, dtype=np.int32)
    instr[:INSTR_WORDS] = _enc_header(
        opcode, si, sj, chi_l, chi_m, chi_r, chi_new, params, off_i, off_j
    )
    gf = gate_mat.ravel().astype(np.complex128)
    for k in range(16):
        instr[INSTR_WORDS + k * 2]     = _f32_bits(gf[k].real)
        instr[INSTR_WORDS + k * 2 + 1] = _f32_bits(gf[k].imag)
    return instr


# ─────────────────────────────────────────────────────────────────────────────
#  Circuit compiler: QuantumCircuit → nearest-neighbour gate list
# ─────────────────────────────────────────────────────────────────────────────

def _qubit_index(circuit, qubit) -> int:
    try:
        return circuit.find_bit(qubit).index
    except Exception:
        return circuit.qubits.index(qubit)


def _decompose_ccx(c0: int, c1: int, tgt: int) -> list:
    return [
        ('h',   [tgt], None),       ('cx', [c1,  tgt], None),
        ('tdg', [tgt], None),       ('cx', [c0,  tgt], None),
        ('t',   [tgt], None),       ('cx', [c1,  tgt], None),
        ('tdg', [tgt], None),       ('cx', [c0,  tgt], None),
        ('t',   [c1],  None),       ('t',  [tgt], None),
        ('h',   [tgt], None),       ('cx', [c0,  c1],  None),
        ('t',   [c0],  None),       ('tdg',[c1],  None),
        ('cx',  [c0,  c1],  None),
    ]


def compile_circuit(circuit, n_qubits: int) -> Tuple[list, list]:
    """
    Compile QuantumCircuit to a nearest-neighbour gate list.

    Returns (gate_list, measured_qubit_indices).

    Each gate dict:
        type      : '1q' | '2q' | 'reset'
        gate_name : str (canonical name)
        sites     : [i] for 1q, [i, i+1] for 2q (always adjacent after routing)
        params    : list[float] | None
        opcode    : int
        gate_mat  : np.ndarray (2×2 or 4×4 complex128)
    """
    perm = list(range(n_qubits))   # perm[logical] = physical site
    gates_out: list = []
    measured: list = []

    def _swap_sites(a: int, b: int):
        sa, sb = min(a, b), max(a, b)
        gates_out.append({
            'type': '2q', 'gate_name': 'swap', 'sites': [sa, sb],
            'params': None, 'opcode': _OP2Q['swap'],
            'gate_mat': GATE_MATRICES['swap'],
        })
        for idx in range(n_qubits):
            if perm[idx] == sa:
                perm[idx] = sb
            elif perm[idx] == sb:
                perm[idx] = sa

    def _route_2q(pi: int, pj: int, nm: str, ps: Optional[list]):
        if pi > pj:
            pi, pj = pj, pi
            pp = np.array([0, 2, 1, 3])
            gm = _gate_2q(nm, ps).reshape(4, 4)[np.ix_(pp, pp)]
        else:
            gm = _gate_2q(nm, ps)
        for k in range(pj, pi + 1, -1):
            _swap_sites(k - 1, k)
        gates_out.append({
            'type': '2q', 'gate_name': nm, 'sites': [pi, pi + 1],
            'params': ps, 'opcode': _OP2Q.get(nm, 17),
            'gate_mat': gm,
        })
        for k in range(pi + 1, pj):
            _swap_sites(k, k + 1)

    for inst in circuit.data:
        nm  = inst.operation.name.lower()
        qs  = [_qubit_index(circuit, q) for q in inst.qubits]
        ps  = [float(p) for p in getattr(inst.operation, 'params', [])] or None

        if nm == 'measure':
            for q in qs:
                if perm[q] not in measured:
                    measured.append(perm[q])
            continue
        if nm in ('barrier', 'snapshot', 'delay'):
            continue
        if nm == 'id':
            continue

        if nm == 'reset':
            gates_out.append({
                'type': 'reset', 'gate_name': 'reset',
                'sites': [perm[qs[0]]], 'params': None,
                'opcode': -1, 'gate_mat': None,
            })
            continue

        if nm in _SINGLE_QUBIT_GATES or nm in _OP1Q:
            site = perm[qs[0]]
            gates_out.append({
                'type': '1q', 'gate_name': nm, 'sites': [site],
                'params': ps, 'opcode': _OP1Q.get(nm, 0),
                'gate_mat': _gate_1q(nm, ps),
            })
            continue

        if nm in _TWO_QUBIT_GATES or nm in _OP2Q:
            pi, pj = perm[qs[0]], perm[qs[1]]
            if abs(pi - pj) == 1:
                si, sj = min(pi, pj), max(pi, pj)
                if pi > pj:
                    pp = np.array([0, 2, 1, 3])
                    gm = _gate_2q(nm, ps).reshape(4, 4)[np.ix_(pp, pp)]
                else:
                    gm = _gate_2q(nm, ps)
                gates_out.append({
                    'type': '2q', 'gate_name': nm, 'sites': [si, sj],
                    'params': ps, 'opcode': _OP2Q.get(nm, 17),
                    'gate_mat': gm,
                })
            else:
                _route_2q(pi, pj, nm, ps)
            continue

        if nm in ('ccx', 'toffoli', 'ccnot'):
            c0, c1, tgt = perm[qs[0]], perm[qs[1]], perm[qs[2]]
            for sub_nm, sub_qs, sub_ps in _decompose_ccx(c0, c1, tgt):
                if sub_nm in _SINGLE_QUBIT_GATES or sub_nm in _OP1Q:
                    gates_out.append({
                        'type': '1q', 'gate_name': sub_nm,
                        'sites': [sub_qs[0]], 'params': sub_ps,
                        'opcode': _OP1Q.get(sub_nm, 0),
                        'gate_mat': _gate_1q(sub_nm, sub_ps),
                    })
                else:
                    pi2, pj2 = sub_qs[0], sub_qs[1]
                    if abs(pi2 - pj2) == 1:
                        si2, sj2 = min(pi2, pj2), max(pi2, pj2)
                        gates_out.append({
                            'type': '2q', 'gate_name': sub_nm,
                            'sites': [si2, sj2], 'params': sub_ps,
                            'opcode': _OP2Q.get(sub_nm, 17),
                            'gate_mat': _gate_2q(sub_nm, sub_ps),
                        })
                    else:
                        _route_2q(pi2, pj2, sub_nm, sub_ps)
            continue

    if not measured:
        measured = list(range(n_qubits))

    return gates_out, sorted(measured)


# ─────────────────────────────────────────────────────────────────────────────
#  Core FPGA execution engine for one circuit on one open device+kernel
# ─────────────────────────────────────────────────────────────────────────────

def _run_circuit_fpga(circuit, shots: int,
                      chi_max: int, svd_cutoff: float,
                      seed: Optional[int],
                      device, kernel,
                      xrt_mod) -> Tuple[MPSState, Dict[str, int], Dict[str, float]]:
    """
    Execute one circuit on an already-initialised FPGA device + kernel.
    Returns (mps_state_after_gates, counts_dict, phase_times).

    Gate execution strategy:
      1q gates    : batched in BRAM gate buffer → one kernel call per batch.
      2q gates (chi ≤ CHI_FULL_SVD)  : OP_2Q_FULL  → kernel does SVD+split.
      2q gates (chi > CHI_FULL_SVD)  : OP_2Q_CONTRACT → kernel contracts+applies;
                                        host reads theta, runs scipy SVD, writes back.

    phase_times (wall-clock seconds, additive perf_counter instrumentation only
    — does not affect control flow, numerics, buffer handling, or allocation):
      fpga_kernel  : kernel(...).wait() calls
      hbm_transfer : host<->device buffer writes/reads (bo.write/bo.sync and
                     the _hbm_read/_hbm_write helpers)
      cpu_svd      : scipy SVD truncation math for 2q gates
      cpu_qr       : QR normalization calls — not invoked on this execution
                     path (MPSState.left/right_normalize_site are unused here),
                     kept at 0.0 for schema completeness
      cpu_sampling : measurement/shot sampling after the gate loop
      host_other   : total wall time minus the five phases above (circuit
                     compile, HBM allocation bookkeeping, Python orchestration)
    """
    _t_total0 = time.perf_counter()
    phase_times: Dict[str, float] = {
        'fpga_kernel': 0.0,
        'hbm_transfer': 0.0,
        'cpu_svd': 0.0,
        'cpu_qr': 0.0,
        'cpu_sampling': 0.0,
    }

    nq  = circuit.num_qubits
    mps = MPSState(nq, chi_max, svd_cutoff)

    # ── Compile circuit first (needed to size BOs for gate-phase allocations) ─
    gate_list, measured_qubits = compile_circuit(circuit, nq)

    # ── HBM allocation ────────────────────────────────────────────────────────
    alloc = _HBMAlloc()
    site_offsets: List[int] = []
    for i in range(nq):
        site_offsets.append(alloc.alloc_site(mps.chi_l(i), mps.chi_r(i)))
    theta_off = alloc.alloc(chi_max * 2 * chi_max * 2)   # workspace
    init_amps = alloc.total

    # Each 2Q gate creates 2 new padded site allocations (worst-case chi_max²).
    max_site_amps = ((chi_max * 2 * chi_max + NUM_HBM_BANKS - 1)
                     // NUM_HBM_BANKS) * NUM_HBM_BANKS
    n_2q = sum(1 for g in gate_list if g['type'] == '2q')
    total_amps = init_amps + n_2q * 2 * max_site_amps

    # ── Allocate HBM BOs (one per bank, interleaved) ─────────────────────────
    amps_per_bank = (total_amps + NUM_HBM_BANKS - 1) // NUM_HBM_BANKS
    hbm_bytes = max(amps_per_bank * 2 * 8, 4096)
    banks: List = [
        xrt_mod.bo(device, hbm_bytes,
                   xrt_mod.bo.flags.normal,
                   kernel.group_id(bid))
        for bid in range(NUM_HBM_BANKS)
    ]
    gate_bo = xrt_mod.bo(device,
                         max(MAX_GATES_BRAM * WORDS_PER_GATE * 4, 4096),
                         xrt_mod.bo.flags.normal,
                         kernel.group_id(8))

    # ── Upload |0...0⟩ state ─────────────────────────────────────────────────
    _t0 = time.perf_counter()
    for i in range(nq):
        _hbm_write(banks, mps.sites[i], site_offsets[i],
                   mps.chi_l(i) * 2 * mps.chi_r(i), xrt_mod)
    phase_times['hbm_transfer'] += time.perf_counter() - _t0

    # ── Execute gate list ─────────────────────────────────────────────────────
    pending_1q: list = []

    def _flush_1q():
        nonlocal pending_1q
        if not pending_1q:
            return
        n_g = len(pending_1q)
        enc = np.concatenate(pending_1q, axis=0).astype(np.int32)

        _t0 = time.perf_counter()
        gate_bo.write(enc.tobytes(), 0)
        gate_bo.sync(xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                     enc.nbytes, 0)
        phase_times['hbm_transfer'] += time.perf_counter() - _t0

        _t0 = time.perf_counter()
        kernel(*banks, gate_bo, n_g, nq, chi_max).wait()
        phase_times['fpga_kernel'] += time.perf_counter() - _t0

        # Read updated site tensors back so CPU MPS state stays coherent
        _t0 = time.perf_counter()
        for instr in pending_1q:
            s     = int(instr[1])
            chl   = mps.chi_l(s)
            chr_  = mps.chi_r(s)
            mps.sites[s] = _hbm_read(
                banks, site_offsets[s],
                chl * 2 * chr_, (chl, 2, chr_), xrt_mod
            )
        phase_times['hbm_transfer'] += time.perf_counter() - _t0
        pending_1q.clear()

    for gate in gate_list:

        if gate['type'] == 'reset':
            _flush_1q()
            s = gate['sites'][0]
            t = mps.sites[s].copy()
            t[:, 1, :] = 0.0
            nm = np.linalg.norm(t)
            if nm > 1e-15:
                t /= nm
            else:
                # Site was fully polarized into the sigma=1 branch (e.g. a
                # bond-dimension-1 site sitting entirely in |1>): zeroing
                # sigma=1 leaves an all-zero tensor with nothing to
                # renormalize. Fall back to a clean |0> product state at
                # this site instead of writing NaNs/zeros downstream,
                # mirroring fpga_simulator_v07.py's _apply_reset fallback.
                t = np.zeros_like(t)
                t[0, 0, 0] = 1.0
            mps.sites[s] = t
            _t0 = time.perf_counter()
            _hbm_write(banks, t, site_offsets[s],
                       mps.chi_l(s) * 2 * mps.chi_r(s), xrt_mod)
            phase_times['hbm_transfer'] += time.perf_counter() - _t0
            continue

        if gate['type'] == '1q':
            s = gate['sites'][0]
            pending_1q.append(_build_1q_instr(
                gate['opcode'], s,
                mps.chi_l(s), mps.chi_r(s),
                gate['params'], site_offsets[s],
                gate['gate_mat']
            ))
            if len(pending_1q) >= MAX_GATES_BRAM - 4:
                _flush_1q()
            continue

        if gate['type'] == '2q':
            _flush_1q()
            si, sj = gate['sites'][0], gate['sites'][1]
            assert sj == si + 1, \
                f"Non-adjacent 2q gate after routing: sites {si},{sj}"

            chi_l = mps.chi_l(si)
            chi_m = mps.chi_r(si)
            chi_r = mps.chi_r(sj)

            # Always OP_2Q_CONTRACT: FPGA contracts+applies gate → writes theta
            # to theta_off; host reads theta, runs scipy SVD, writes back A_i/A_j.
            # word [6] carries theta_off so the kernel knows where to write.
            instr = _build_2q_instr(
                OP_2Q_CONTRACT,
                si, sj, chi_l, chi_m, chi_r,
                theta_off,    # word [6] = HBM theta workspace offset
                gate['params'], site_offsets[si], site_offsets[sj],
                gate['gate_mat']
            ).reshape(1, WORDS_PER_GATE).astype(np.int32)

            _t0 = time.perf_counter()
            gate_bo.write(instr.tobytes(), 0)
            gate_bo.sync(xrt_mod.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
                         instr.nbytes, 0)
            phase_times['hbm_transfer'] += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            kernel(*banks, gate_bo, 1, nq, chi_max).wait()
            phase_times['fpga_kernel'] += time.perf_counter() - _t0

            # Host SVD: read theta from workspace, truncate, write A_i / A_j back
            _t0 = time.perf_counter()
            theta = _hbm_read(banks, theta_off,
                              chi_l * 4 * chi_r,
                              (chi_l * 2, 2 * chi_r), xrt_mod)
            phase_times['hbm_transfer'] += time.perf_counter() - _t0

            _t0 = time.perf_counter()
            U, S, Vh = la.svd(theta, full_matrices=False)
            thresh = svd_cutoff * S[0] if len(S) > 0 else 0.0
            keep = int(np.clip(np.sum(S > thresh), 1, chi_max))
            if keep < len(S):
                dropped_norm_sq = np.sum(S[keep:] ** 2)
                if dropped_norm_sq > 0:
                    mps.norm_log += 0.5 * math.log(
                        1.0 - dropped_norm_sq / max(np.sum(S ** 2), 1e-300)
                    )
            U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
            nm_ = np.linalg.norm(S)
            if nm_ > 1e-15:
                mps.norm_log += math.log(nm_)
                S /= nm_
            sqS = np.sqrt(S)
            chi_new = keep
            phase_times['cpu_svd'] += time.perf_counter() - _t0

            mps.sites[si] = (U * sqS).reshape(chi_l, 2, chi_new)
            mps.sites[sj] = (sqS[:, None] * Vh).reshape(chi_new, 2, chi_r)

            site_offsets[si] = alloc.alloc_site(chi_l, chi_new)
            site_offsets[sj] = alloc.alloc_site(chi_new, chi_r)

            _t0 = time.perf_counter()
            _hbm_write(banks, mps.sites[si], site_offsets[si],
                       chi_l * 2 * chi_new, xrt_mod)
            _hbm_write(banks, mps.sites[sj], site_offsets[sj],
                       chi_new * 2 * chi_r, xrt_mod)
            phase_times['hbm_transfer'] += time.perf_counter() - _t0
            continue

    _flush_1q()

    # ── Sequential conditional sampling on CPU ────────────────────────────────
    _t0 = time.perf_counter()
    rng = np.random.default_rng(seed)
    counts = mps.get_counts(shots, measured_qubits, rng)
    phase_times['cpu_sampling'] += time.perf_counter() - _t0

    banks.clear()

    _t_total = time.perf_counter() - _t_total0
    phase_times['host_other'] = max(0.0, _t_total - (
        phase_times['fpga_kernel'] + phase_times['hbm_transfer']
        + phase_times['cpu_svd'] + phase_times['cpu_qr']
        + phase_times['cpu_sampling']
    ))

    return mps, counts, phase_times


# ─────────────────────────────────────────────────────────────────────────────
#  Worker process — Mode A (shot-parallel, one card)
# ─────────────────────────────────────────────────────────────────────────────

def _worker_a(card_id: int, xclbin: str, kernel_name: str,
              cmd_q: Queue, res_q: Queue,
              ready: Event, shutdown: Event):
    """Persistent worker for the single FPGA card at device index 0."""
    if card_id != 0:
        raise RuntimeError(
            f'Single-card verification build: device index must be 0, got {card_id}.'
        )
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    _device = None
    _kernel = None
    _fpga_ready = False

    def _init_fpga():
        nonlocal _device, _kernel, _fpga_ready
        import pyxrt as _xrt
        _device = _xrt.device(card_id)
        uuid    = _device.load_xclbin(xclbin)
        _kernel = _xrt.kernel(_device, uuid, kernel_name)
        _fpga_ready = True
        logger.info('Device %d: FPGA ready (xclbin UUID %s)', card_id, uuid)

    try:
        ready.set()
        while not shutdown.is_set():
            try:
                cmd = cmd_q.get(timeout=0.25)
            except Exception:
                continue
            if cmd is None:
                break

            ct = cmd[0]

            if ct == 'ping':
                res_q.put(('pong', card_id))
                continue

            if ct == 'run_fpga':
                import pickle, pyxrt as _xrt
                circuit_b, shots, chi_max, svd_cutoff, seed = cmd[1:]
                if not _fpga_ready:
                    _init_fpga()
                circuit = pickle.loads(circuit_b)
                try:
                    mps_st, counts, phase_times = _run_circuit_fpga(
                        circuit, shots, chi_max, svd_cutoff, seed,
                        _device, _kernel, _xrt
                    )
                    sv = None
                    if circuit.num_qubits <= 24:
                        try:
                            sv = mps_st.get_statevector()
                        except Exception:
                            pass
                    res_q.put(('done', card_id, counts, sv,
                               mps_st.max_bond(), mps_st.bond_dims(),
                               phase_times))
                except Exception as e:
                    res_q.put(('error', card_id, str(e)))
                continue

    except Exception as e:
        logger.error('Worker card %d fatal: %s', card_id, e)
        res_q.put(('error', card_id, str(e)))


# ─────────────────────────────────────────────────────────────────────────────
#  FPGAMPSResult
# ─────────────────────────────────────────────────────────────────────────────

class FPGAMPSResult:
    """Qiskit-compatible result container."""

    def __init__(self, results: list, shots: int,
                 num_cards: int, elapsed: float, mode: str):
        self._results = results
        self.shots     = shots
        self.num_cards = num_cards
        self.total_time = elapsed
        self.mode      = mode

    def get_counts(self, idx: int = 0) -> Dict[str, int]:
        return self._results[idx]['counts']

    def get_statevector(self, idx: int = 0) -> np.ndarray:
        sv = self._results[idx].get('statevector')
        if sv is None:
            raise ValueError(
                'Statevector not available (n > 24 or not computed).'
            )
        return sv

    def get_mps(self, idx: int = 0) -> Optional[MPSState]:
        return self._results[idx].get('mps')

    def get_max_bond(self, idx: int = 0) -> int:
        return self._results[idx].get('max_bond', -1)

    def get_bond_dims(self, idx: int = 0) -> List[int]:
        return self._results[idx].get('bond_dims', [])

    def get_phase_times(self, idx: int = 0) -> Dict[str, float]:
        """Per-phase wall-clock seconds from the additive perf_counter
        instrumentation in _run_circuit_fpga (fpga_kernel, hbm_transfer,
        cpu_svd, cpu_qr, cpu_sampling, host_other)."""
        return self._results[idx].get('phase_times', {})

    def success(self) -> bool:
        return True


# ─────────────────────────────────────────────────────────────────────────────
#  FPGAMPSSimulator — main class
# ─────────────────────────────────────────────────────────────────────────────

class FPGAMPSSimulator:
    """
    FPGA-accelerated Matrix Product State quantum circuit simulator.

    Single-card verification build: exactly one Alveo U55C at device index 0.
    There is no CPU engine and no multi-card mode. Any condition that would
    previously have fallen back to CPU simulation instead raises RuntimeError.

    Parameters
    ----------
    num_cards   : must be 1. Any other value raises ValueError.
    chi_max     : Maximum bond dimension. Default 64, and must not exceed 64
                  (CHI_MAX_SVD_HW) -- the kernel's on-chip theta workspace is
                  fixed at that size and silently truncates/drops contraction
                  terms above it with no error signalled back to the host, so
                  this is enforced here rather than left to corrupt results.
    svd_cutoff  : Relative SVD truncation threshold. Default 1e-12.
    engine      : must be 'fpga'. Any other value raises ValueError.
                  RuntimeError is raised if XRT is unavailable, device index 0
                  cannot be opened, or the xclbin cannot be loaded.
    mode        : must be 'shots'. Any other value raises ValueError.
    xclbin_path : Path to .xclbin; None → auto-search cwd and script dir.
    """

    def __init__(self,
                 num_cards: int = NUM_CARDS,
                 chi_max: int = CHI_MAX_SVD_HW,
                 svd_cutoff: float = 1e-12,
                 engine: str = 'fpga',
                 mode: str = 'shots',
                 xclbin_path: Optional[str] = None):

        if num_cards != 1:
            raise ValueError(
                f'Single-card verification build: num_cards must be 1, got {num_cards}. '
                'Multi-card operation is not permitted in this environment.'
            )
        if mode != 'shots':
            raise ValueError(
                f"Single-card verification build: mode must be 'shots', got {mode!r}. "
                "Distributed/multi-card mode has been removed from this environment."
            )
        if engine != 'fpga':
            raise ValueError(
                f"Single-card verification build: engine must be 'fpga', got {engine!r}. "
                "There is no CPU engine in this environment — CPU fallback is forbidden."
            )
        if not (1 <= chi_max <= CHI_MAX_SVD_HW):
            raise ValueError(
                f'chi_max must be in [1, {CHI_MAX_SVD_HW}], got {chi_max}. The FPGA '
                f'kernel\'s theta workspace is fixed at CHI_MAX_SVD={CHI_MAX_SVD_HW} '
                'and silently drops contraction terms above it with no runtime '
                'error, so a larger chi_max would corrupt results rather than '
                'raise -- this is rejected here instead.'
            )

        self.num_cards  = num_cards
        self.chi_max    = chi_max
        self.svd_cutoff = svd_cutoff
        self.mode       = mode

        # ── Resolve engine — FPGA only, hard failure otherwise ────────────────
        if not XRT_AVAILABLE:
            raise RuntimeError(
                "engine='fpga' requires pyxrt, which is not importable. "
                'Source /opt/xilinx/xrt/setup.sh and retry. '
                'No CPU fallback is permitted in this environment.'
            )
        try:
            xrt.device(0)
        except Exception as e:
            raise RuntimeError(
                f'FPGA path unavailable: could not open device index 0 ({e}). '
                'No CPU fallback is permitted in this environment.'
            ) from e
        self._engine = 'fpga'

        # ── Resolve xclbin path ───────────────────────────────────────────────
        if xclbin_path is None:
            xclbin_path = os.environ.get('FPGA_MPS_XCLBIN')
        if xclbin_path is None:
            for candidate in [
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             DEFAULT_XCLBIN),
                os.path.join(os.getcwd(), DEFAULT_XCLBIN),
            ]:
                if os.path.isfile(candidate):
                    xclbin_path = candidate
                    break
            if xclbin_path is None:
                raise RuntimeError(
                    f"FPGA path unavailable: '{DEFAULT_XCLBIN}' not found. "
                    "Pass xclbin_path= explicitly or set FPGA_MPS_XCLBIN. "
                    'No CPU fallback is permitted in this environment.'
                )
        if not os.path.isfile(xclbin_path):
            raise RuntimeError(
                f'FPGA path unavailable: xclbin not found at {xclbin_path!r}. '
                'No CPU fallback is permitted in this environment.'
            )
        self._xclbin = os.path.abspath(xclbin_path)
        self._kname  = self._probe_kernel_name()

        # ── Worker state ──────────────────────────────────────────────────────
        self._workers:      List[Process] = []
        self._cmd_qs:       List[Queue]   = []
        self._res_q:        Optional[Queue] = None
        self._ready_evts:   List[Event]   = []
        self._shutdown_evts: List[Event]  = []
        self._spawned       = False
        self._spawned_mode: Optional[str] = None

        atexit.register(self.shutdown)

    # ─────────────────────────────────────────────────────────────────────────
    #  Kernel name probing
    # ─────────────────────────────────────────────────────────────────────────

    def _probe_kernel_name(self) -> str:
        try:
            d    = xrt.device(0)
            uuid = d.load_xclbin(self._xclbin)
            for kn in (KERNEL_NAME, 'mps_kernel', 'quantum_mps'):
                try:
                    xrt.kernel(d, uuid, kn)
                    return kn
                except Exception:
                    pass
        except Exception as e:
            logger.warning('Kernel probe failed: %s — using default name', e)
        return KERNEL_NAME

    # ─────────────────────────────────────────────────────────────────────────
    #  Worker spawn / teardown
    # ─────────────────────────────────────────────────────────────────────────

    def _ensure_workers(self, eff_mode: str):
        assert eff_mode == 'shots', f'Only mode="shots" is supported, got {eff_mode!r}.'
        if self._spawned and self._spawned_mode == eff_mode:
            if all(w.is_alive() for w in self._workers):
                return
            logger.warning('Dead worker detected; respawning.')
        if self._spawned:
            self._teardown()

        self._res_q = Queue()

        # Single card only: device index 0.
        cid = 0
        cq, re, se = Queue(), Event(), Event()
        w = Process(
            target=_worker_a,
            args=(cid, self._xclbin or '', self._kname or '',
                  cq, self._res_q, re, se),
            daemon=True
        )
        w.start()
        self._workers.append(w)
        self._cmd_qs.append(cq)
        self._ready_evts.append(re)
        self._shutdown_evts.append(se)

        t0 = time.time()
        for i, ev in enumerate(self._ready_evts):
            ev.wait(timeout=max(1.0, 60.0 - (time.time() - t0)))

        self._spawned      = True
        self._spawned_mode = eff_mode

    def _teardown(self):
        for se in self._shutdown_evts:
            try:
                se.set()
            except Exception:
                pass
        for cq in self._cmd_qs:
            try:
                cq.put_nowait(None)
            except Exception:
                pass
        for w in self._workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()
                w.join(timeout=2)
        self._workers.clear()
        self._cmd_qs.clear()
        self._ready_evts.clear()
        self._shutdown_evts.clear()
        try:
            if self._res_q:
                while not self._res_q.empty():
                    self._res_q.get_nowait()
        except Exception:
            pass
        self._res_q        = None
        self._spawned      = False
        self._spawned_mode = None

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    def run(self,
            circuits,
            shots: int = 1024,
            seed: Optional[int] = None) -> FPGAMPSResult:
        """
        Run one or more QuantumCircuits.

        Parameters
        ----------
        circuits : QuantumCircuit or list[QuantumCircuit]
        shots    : measurement shots per circuit
        seed     : RNG seed for reproducibility

        Returns
        -------
        FPGAMPSResult
        """
        if not isinstance(circuits, list):
            circuits = [circuits]

        t0 = time.time()

        self._ensure_workers('shots')
        return self._run_shots(circuits, shots, seed, t0)

    # ─────────────────────────────────────────────────────────────────────────
    #  FPGA shot-parallel path (single card, device index 0)
    # ─────────────────────────────────────────────────────────────────────────

    def _run_shots(self, circuits, shots, seed, t0) -> FPGAMPSResult:
        import pickle
        cmd_type    = 'run_fpga'
        tasks       = list(enumerate(circuits))
        pending:    List[Tuple[int, int]] = []
        free_cards  = list(range(self.num_cards))
        results_map: Dict[int, dict] = {}
        remaining   = list(range(len(tasks)))

        while remaining or pending:
            while remaining and free_cards:
                ti  = remaining.pop(0)
                cid = free_cards.pop(0)
                _, circuit = tasks[ti]
                s = (seed + ti) if seed is not None else None
                self._cmd_qs[cid].put(
                    (cmd_type,
                     pickle.dumps(circuit),
                     shots, self.chi_max, self.svd_cutoff, s)
                )
                pending.append((ti, cid))

            if pending:
                r = self._res_q.get(timeout=600)
                if r[0] == 'done':
                    _, cid, counts, sv, max_b, bdims, phase_times = r
                    ti = next(i for i, c in pending if c == cid)
                    pending.remove((ti, cid))
                    free_cards.append(cid)
                    results_map[ti] = {
                        'counts': counts, 'statevector': sv,
                        'max_bond': max_b, 'bond_dims': bdims,
                        'phase_times': phase_times,
                    }
                elif r[0] == 'error':
                    raise RuntimeError(f'Card {r[1]} error: {r[2]}')

        results = [results_map[i] for i in range(len(tasks))]
        return FPGAMPSResult(results, shots, self.num_cards,
                             time.time() - t0, 'shots_fpga')

    # ─────────────────────────────────────────────────────────────────────────
    #  Introspection / lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    def get_card_stats(self) -> List[dict]:
        return [
            {
                'card_id':        i,
                'pid':            self._workers[i].pid if i < len(self._workers) else None,
                'alive':          (self._workers[i].is_alive()
                                   if i < len(self._workers) else False),
                'engine':         self._engine.upper(),
                'execution_mode': self._engine.upper(),
                'xclbin':         self._xclbin,
            }
            for i in range(self.num_cards)
        ]

    def shutdown(self):
        if self._spawned:
            self._teardown()

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.shutdown()
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  Pauli expectation value helper
# ─────────────────────────────────────────────────────────────────────────────

def _pauli_expectation_sv(sv: np.ndarray, pauli: str) -> float:
    """Compute ⟨ψ|P|ψ⟩ for a Pauli string P (e.g. 'ZZ', 'IXZ')."""
    n   = int(round(math.log2(len(sv))))
    psi = sv.copy()
    for i, p in enumerate(pauli.upper()):
        if p == 'I':
            continue
        qubit  = n - 1 - i
        mat    = GATE_MATRICES.get(p.lower(), np.eye(2, dtype=np.complex128))
        N      = 1 << n
        stride = 1 << qubit
        for j in range(0, N, stride * 2):
            for k in range(stride):
                a0 = psi[j + k]
                a1 = psi[j + k + stride]
                psi[j + k]          = mat[0, 0] * a0 + mat[0, 1] * a1
                psi[j + k + stride] = mat[1, 0] * a0 + mat[1, 1] * a1
    return float(np.real(np.dot(sv.conj(), psi)))


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point / self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('FPGA MPS Simulator — Host Driver')
    print('=' * 56)
    print(f'  XRT Available : {XRT_AVAILABLE}')
    print(f'  Qiskit        : {QISKIT_AVAILABLE}')
    print(f'  Kernel name   : {KERNEL_NAME}')
    print(f'  Default xclbin: {DEFAULT_XCLBIN}')
    print(f'  HBM banks     : {NUM_HBM_BANKS}  (+1 gate = 9 AXI masters total)')
    print(f'  FPGA SVD cap  : chi ≤ {CHI_FULL_SVD}')
    print()

    if XRT_AVAILABLE:
        try:
            d  = xrt.device(0)
            print(f'  Device 0 BDF  : {d.get_info(xrt.xrt_info_device.bdf)}')
            xb = DEFAULT_XCLBIN
            if os.path.isfile(xb):
                uuid = d.load_xclbin(xb)
                k    = xrt.kernel(d, uuid, KERNEL_NAME)
                print(f'  xclbin loaded : {xb}')
                print(f'  Kernel OK     : {KERNEL_NAME}')
            else:
                print(f'  xclbin        : NOT FOUND — run "make xclbin" first')
        except Exception as e:
            print(f'  Device probe  : FAILED — {e}')
    print()
    print('Usage (single-card verification build — engine="fpga" only):')
    print('  from fpga_mps_simulator import FPGAMPSSimulator')
    print('  sim = FPGAMPSSimulator(num_cards=1, engine="fpga", mode="shots")')
    print('  result = sim.run(circuit, shots=1024)')
    print('  print(result.get_counts())')

    if QISKIT_AVAILABLE:
        print()
        print('Self-test (FPGA device 0, 3-qubit GHZ) — no CPU fallback:')
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(1, 2)
        qc.measure_all()
        try:
            sim = FPGAMPSSimulator(num_cards=1, engine='fpga', mode='shots', chi_max=16)
            res = sim.run(qc, shots=1024, seed=0)
            assert res.mode == 'shots_fpga', f'Expected mode=shots_fpga, got {res.mode!r}'
            counts = res.get_counts()
            total  = sum(counts.values())
            p000   = counts.get('000', 0) / total
            p111   = counts.get('111', 0) / total
            ok     = p000 + p111 > 0.95
            print(f'  mode          : {res.mode}')
            print(f'  P(000)+P(111) = {p000+p111:.3f}  {"PASS" if ok else "FAIL"}')
            sim.shutdown()
        except RuntimeError as e:
            print(f'  Self-test SKIPPED — FPGA path unavailable: {e}')
