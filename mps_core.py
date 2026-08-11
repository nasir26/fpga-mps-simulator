"""
Author: Nasir Ali
Organization: C-DAC Noida
Date: June 2026
Target: All platforms (pure NumPy/SciPy, no hardware dependencies)
"""

from __future__ import annotations
import math
import logging
import numpy as np
import scipy.linalg as la
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Gate matrices (complex128)
# ─────────────────────────────────────────────────────────────────────────────

_S2 = math.sqrt(2)
_IS2 = 1.0 / _S2

GATE_MATRICES: Dict[str, np.ndarray] = {
    'id':   np.eye(2, dtype=np.complex128),
    'h':    np.array([[_IS2, _IS2], [_IS2, -_IS2]], dtype=np.complex128),
    'x':    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'y':    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'z':    np.array([[1, 0], [0, -1]], dtype=np.complex128),
    's':    np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    'sdg':  np.array([[1, 0], [0, -1j]], dtype=np.complex128),
    't':    np.array([[1, 0], [0, np.exp(1j*math.pi/4)]], dtype=np.complex128),
    'tdg':  np.array([[1, 0], [0, np.exp(-1j*math.pi/4)]], dtype=np.complex128),
    'sx':   np.array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]], dtype=np.complex128),
    'sxdg': np.array([[0.5-0.5j, 0.5+0.5j], [0.5+0.5j, 0.5-0.5j]], dtype=np.complex128),
    # 2-qubit gates (4×4, row-major order |00⟩,|01⟩,|10⟩,|11⟩)
    'cx':   np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex128),
    'cz':   np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1]], dtype=np.complex128),
    'cy':   np.array([[1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0]], dtype=np.complex128),
    'swap': np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex128),
    'iswap':np.array([[1,0,0,0],[0,0,1j,0],[0,1j,0,0],[0,0,0,1]], dtype=np.complex128),
    'dcx':  np.array([[1,0,0,0],[0,0,0,1],[0,1,0,0],[0,0,1,0]], dtype=np.complex128),
}


def _gate_1q(name: str, params: Optional[List[float]]) -> np.ndarray:
    """Return 2×2 complex128 gate matrix."""
    nm = name.lower()
    if nm in GATE_MATRICES:
        return GATE_MATRICES[nm]
    p = [float(x) for x in (params or [])]
    if nm == 'rx':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        return np.array([[c, -1j*s], [-1j*s, c]], dtype=np.complex128)
    if nm == 'ry':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if nm == 'rz':
        e = np.exp(1j * p[0] / 2)
        return np.array([[e.conj(), 0], [0, e]], dtype=np.complex128)
    if nm in ('p', 'phase', 'u1'):
        return np.array([[1, 0], [0, np.exp(1j*p[0])]], dtype=np.complex128)
    if nm == 'u2':
        inv_s2 = 1.0 / math.sqrt(2)
        phi, lam = p[0], p[1]
        return inv_s2 * np.array([
            [1, -np.exp(1j*lam)],
            [np.exp(1j*phi), np.exp(1j*(phi+lam))]
        ], dtype=np.complex128)
    if nm in ('u3', 'u'):
        theta, phi, lam = p[0], p[1], p[2]
        ct, st = math.cos(theta/2), math.sin(theta/2)
        return np.array([
            [ct, -np.exp(1j*lam)*st],
            [np.exp(1j*phi)*st, np.exp(1j*(phi+lam))*ct]
        ], dtype=np.complex128)
    raise ValueError(f"Unknown 1q gate: {name}")


def _gate_2q(name: str, params: Optional[List[float]]) -> np.ndarray:
    """Return 4×4 complex128 gate matrix (basis order: |00⟩,|01⟩,|10⟩,|11⟩)."""
    nm = name.lower()
    if nm in GATE_MATRICES:
        return GATE_MATRICES[nm]
    p = [float(x) for x in (params or [])]
    if nm in ('cp', 'cphase'):
        m = np.eye(4, dtype=np.complex128)
        m[3, 3] = np.exp(1j*p[0])
        return m
    if nm == 'ch':
        inv_s2 = 1.0 / math.sqrt(2)
        m = np.eye(4, dtype=np.complex128)
        m[2:, 2:] = inv_s2 * np.array([[1, 1], [1, -1]])
        return m
    if nm == 'crx':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        m = np.eye(4, dtype=np.complex128)
        m[2:, 2:] = np.array([[c, -1j*s], [-1j*s, c]])
        return m
    if nm == 'cry':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        m = np.eye(4, dtype=np.complex128)
        m[2:, 2:] = np.array([[c, -s], [s, c]])
        return m
    if nm == 'crz':
        e = np.exp(1j * p[0] / 2)
        m = np.eye(4, dtype=np.complex128)
        m[2, 2] = e.conj()
        m[3, 3] = e
        return m
    if nm == 'csx':
        sx = np.array([[0.5+0.5j, 0.5-0.5j], [0.5-0.5j, 0.5+0.5j]], dtype=np.complex128)
        m = np.eye(4, dtype=np.complex128)
        m[2:, 2:] = sx
        return m
    if nm == 'ecr':
        inv_s2 = 1.0 / math.sqrt(2)
        return inv_s2 * np.array([
            [0, 0, 1, 1j],
            [0, 0, 1j, 1],
            [1, -1j, 0, 0],
            [-1j, 1, 0, 0],
        ], dtype=np.complex128)
    if nm == 'rxx':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        return np.array([
            [c, 0, 0, -1j*s],
            [0, c, -1j*s, 0],
            [0, -1j*s, c, 0],
            [-1j*s, 0, 0, c],
        ], dtype=np.complex128)
    if nm == 'ryy':
        c, s = math.cos(p[0]/2), math.sin(p[0]/2)
        return np.array([
            [c, 0, 0, 1j*s],
            [0, c, -1j*s, 0],
            [0, -1j*s, c, 0],
            [1j*s, 0, 0, c],
        ], dtype=np.complex128)
    if nm == 'rzz':
        ep = np.exp(1j*p[0]/2)
        em = ep.conj()
        return np.diag([em, ep, ep, em])
    raise ValueError(f"Unknown 2q gate: {name}")


# ─────────────────────────────────────────────────────────────────────────────
#  SWAP routing: decompose non-adjacent 2q gate into nearest-neighbour ops
# ─────────────────────────────────────────────────────────────────────────────

def _swap_route(i: int, j: int, gate_name: str, params: Optional[List[float]]):
    """
    Route a 2q gate on (i, j) with |i-j| > 1 to a sequence of nearest-neighbour
    gates. Uses SWAP-chain to bring qubit j adjacent to i, apply gate, SWAP back.
    Returns list of (gate_name, [q0, q1], params) tuples, all nearest-neighbour.
    """
    if j < i:
        i, j = j, i
    ops = []
    for k in range(j, i+1, -1):
        ops.append(('swap', [k-1, k], None))
    ops.append((gate_name, [i, i+1], params))
    for k in range(i+1, j):
        ops.append(('swap', [k, k+1], None))
    return ops


# ─────────────────────────────────────────────────────────────────────────────
#  MPS State
# ─────────────────────────────────────────────────────────────────────────────

class MPSState:
    """
    Matrix Product State for n qubits.

    Attributes
    ----------
    n          : number of qubits
    chi_max    : maximum bond dimension
    svd_cutoff : singular value cutoff (relative to largest S value)
    sites      : list of n tensors, sites[i].shape = (chi_l, 2, chi_r)
    norm_log   : log of the overall norm (accumulated truncation weight)
    """

    def __init__(self, n: int, chi_max: int = 256, svd_cutoff: float = 1e-12):
        self.n = n
        self.chi_max = chi_max
        self.svd_cutoff = svd_cutoff
        self.norm_log: float = 0.0
        # Initialize to |0...0⟩: all sites are [[1,0]] shaped correctly
        self.sites: List[np.ndarray] = []
        for _ in range(n):
            t = np.zeros((1, 2, 1), dtype=np.complex128)
            t[0, 0, 0] = 1.0  # σ=0 (|0⟩ component)
            self.sites.append(t)

    def copy(self) -> 'MPSState':
        s = MPSState(self.n, self.chi_max, self.svd_cutoff)
        s.sites = [t.copy() for t in self.sites]
        s.norm_log = self.norm_log
        return s

    # ── Site-tensor access helpers ───────────────────────────────────────────

    def chi_l(self, i: int) -> int:
        return self.sites[i].shape[0]

    def chi_r(self, i: int) -> int:
        return self.sites[i].shape[2]

    def max_bond(self) -> int:
        if self.n <= 1:
            return 1
        return max(self.chi_r(i) for i in range(self.n - 1))

    def bond_dims(self) -> List[int]:
        return [self.chi_r(i) for i in range(self.n - 1)]

    # ── Canonical form helpers (CPU-side QR normalization) ───────────────────

    def left_normalize_site(self, i: int):
        """
        Left-normalize site i via QR. Updates sites[i] and absorbs R into sites[i+1].
        After: Σ_σ A†[σ] A[σ] = I.
        """
        A = self.sites[i]
        chi_l, d, chi_r = A.shape
        M = A.reshape(chi_l * d, chi_r)
        Q, R = np.linalg.qr(M)
        chi_new = Q.shape[1]
        self.sites[i] = Q.reshape(chi_l, d, chi_new)
        if i + 1 < self.n:
            B = self.sites[i + 1]
            self.sites[i + 1] = np.tensordot(R, B, axes=([1], [0]))

    def right_normalize_site(self, i: int):
        """
        Right-normalize site i via LQ. Updates sites[i] and absorbs L into sites[i-1].
        After: Σ_σ B[σ] B†[σ] = I.
        """
        A = self.sites[i]
        chi_l, d, chi_r = A.shape
        M = A.reshape(chi_l, d * chi_r)
        Q, R = np.linalg.qr(M.T.conj())
        L = R.T.conj()
        B = Q.T.conj()
        chi_new = B.shape[0]
        self.sites[i] = B.reshape(chi_new, d, chi_r)
        if i - 1 >= 0:
            prev = self.sites[i - 1]
            self.sites[i - 1] = np.tensordot(prev, L, axes=([2], [0]))

    def left_canonicalize(self):
        """Left-canonicalize the full chain (sites 0..n-2)."""
        for i in range(self.n - 1):
            self.left_normalize_site(i)

    def right_canonicalize(self):
        """Right-canonicalize the full chain (sites 1..n-1)."""
        for i in range(self.n - 1, 0, -1):
            self.right_normalize_site(i)

    # ── 1-qubit gate application ─────────────────────────────────────────────
    #    (used only for local bookkeeping — bulk gate/tensor contraction for
    #    circuit execution is performed on the FPGA via _run_circuit_fpga)

    def apply_1q(self, site: int, gate: np.ndarray):
        """
        Apply 2×2 unitary gate to qubit `site`.
        A'[χ_l, σ', χ_r] = Σ_σ G[σ', σ] · A[χ_l, σ, χ_r]
        """
        A = self.sites[site]  # (chi_l, 2, chi_r)
        self.sites[site] = np.tensordot(gate, A, axes=([1], [1])).transpose(1, 0, 2)

    # ── SVD truncation (CPU-side, scipy/LAPACK) ──────────────────────────────

    def _svd_truncate(self, M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        SVD of matrix M with chi_max and svd_cutoff truncation.
        Returns (U, sqrt_S, Vh) where S is normalized so ||ψ||=1 is preserved
        up to truncation error. Accumulates log(norm) into self.norm_log.
        """
        U, S, Vh = la.svd(M, full_matrices=False)
        threshold = self.svd_cutoff * S[0] if len(S) > 0 else 0.0
        keep = np.sum(S > threshold)
        keep = int(np.clip(keep, 1, self.chi_max))
        if keep < len(S):
            dropped_norm_sq = np.sum(S[keep:] ** 2)
            if dropped_norm_sq > 0:
                self.norm_log += 0.5 * math.log(1.0 - dropped_norm_sq /
                                                  max(np.sum(S**2), 1e-300))
        U = U[:, :keep]
        S_t = S[:keep]
        Vh = Vh[:keep, :]
        norm = np.linalg.norm(S_t)
        if norm > 1e-15:
            self.norm_log += math.log(norm)
            S_t = S_t / norm
        sqrt_S = np.sqrt(S_t)
        return U, sqrt_S, Vh

    # ── 2-qubit gate application ─────────────────────────────────────────────

    def apply_2q(self, i: int, j: int, gate: np.ndarray):
        """
        Apply 4×4 gate to qubits i and j.
        If |i-j| > 1, SWAP-routes the gate to nearest-neighbour.
        Gate matrix convention: rows/cols ordered as |00⟩,|01⟩,|10⟩,|11⟩.
        """
        if abs(i - j) != 1:
            ops = _swap_route(i, j, '__gate2q__', None)
            for op_name, op_qubits, op_params in ops:
                if op_name == 'swap':
                    self.apply_2q(op_qubits[0], op_qubits[1], GATE_MATRICES['swap'])
                else:
                    self.apply_2q(op_qubits[0], op_qubits[1], gate)
            return

        if i > j:
            i, j = j, i
            perm = np.array([0, 2, 1, 3])
            gate = gate[np.ix_(perm, perm)]

        self._apply_adjacent_2q(i, gate)

    def _apply_adjacent_2q(self, i: int, gate: np.ndarray):
        """
        Apply 4×4 gate to adjacent sites i and i+1.
        Implements contract → apply → SVD → split.
        """
        j = i + 1
        A = self.sites[i]   # (chi_l, 2, chi_m)
        B = self.sites[j]   # (chi_m, 2, chi_r)
        chi_l = A.shape[0]
        chi_m = A.shape[2]
        chi_r = B.shape[2]

        Theta = np.tensordot(A, B, axes=([2], [0]))  # (chi_l, 2, 2, chi_r)

        G = gate.reshape(4, 4)
        Theta_flat = Theta.reshape(chi_l, 4, chi_r)
        Theta_new = np.einsum('ij,kjl->kil', G, Theta_flat)  # (chi_l, 4, chi_r)

        M = Theta_new.reshape(chi_l * 2, 2 * chi_r)

        U, sqrt_S, Vh = self._svd_truncate(M)
        chi_new = len(sqrt_S)

        self.sites[i] = (U * sqrt_S).reshape(chi_l, 2, chi_new)
        self.sites[j] = (sqrt_S[:, None] * Vh).reshape(chi_new, 2, chi_r)

    # ── Measurement / shot sampling (CPU-side, by design) ────────────────────

    def _right_environments(self) -> List[np.ndarray]:
        """
        Compute right environments R[i] for sites i = n-1 downto 0.
        Used for sampling: P(σ_i | σ_0...σ_{i-1}) = Tr[L[i] Σ_σ A[σ] R[i+1] A†[σ]]
        """
        R_list = [None] * (self.n + 1)
        R_list[self.n] = np.array([[1.0]], dtype=np.complex128)
        for i in range(self.n - 1, -1, -1):
            A = self.sites[i]  # (chi_l, 2, chi_r)
            R = R_list[i + 1]  # (chi_r, chi_r)
            R_new = np.einsum('isb,bc,jsc->ij', A, R, A.conj())
            tr = np.real(np.trace(R_new))
            if tr > 1e-300:
                R_new /= tr
            R_list[i] = R_new
        return R_list

    def sample(self, num_shots: int = 1, measured_qubits: Optional[List[int]] = None,
               rng: Optional[np.random.Generator] = None) -> List[str]:
        """
        Sample measurement outcomes using sequential conditional sampling.
        O(n · chi²) per shot. Does NOT reconstruct the full statevector.

        Returns list of bitstrings (ordered msb..lsb matching Qiskit convention:
        bitstring[0] = qubit n-1, bitstring[-1] = qubit 0).
        """
        if rng is None:
            rng = np.random.default_rng()
        if measured_qubits is None:
            measured_qubits = list(range(self.n))

        R_list = self._right_environments()
        outcomes = []

        for _ in range(num_shots):
            v = np.array([1.0], dtype=np.complex128)   # shape (chi_l_0,) = (1,)
            bits = ['0'] * self.n

            for i in range(self.n):
                A = self.sites[i]  # (chi_l, 2, chi_r)
                R = R_list[i + 1]  # (chi_r, chi_r)
                probs = np.zeros(2, dtype=np.float64)
                for sigma in range(2):
                    Av = v @ A[:, sigma, :]
                    probs[sigma] = np.real(Av @ R @ Av.conj())
                probs = np.maximum(probs, 0.0)
                total = probs.sum()
                if total < 1e-15:
                    probs = np.array([0.5, 0.5])
                else:
                    probs /= total

                outcome = rng.choice(2, p=probs)
                bits[i] = str(outcome)

                v = v @ A[:, outcome, :]
                norm = np.linalg.norm(v)
                if norm > 1e-15:
                    v /= norm

            bitstring = ''.join(reversed(bits))
            if set(measured_qubits) != set(range(self.n)):
                ms = sorted(measured_qubits, reverse=True)
                bitstring = ''.join(bits[self.n - 1 - q] for q in ms)
            outcomes.append(bitstring)

        return outcomes

    def get_counts(self, num_shots: int, measured_qubits: Optional[List[int]] = None,
                   rng: Optional[np.random.Generator] = None) -> Dict[str, int]:
        """Sample and return counts dict."""
        samples = self.sample(num_shots, measured_qubits, rng)
        counts: Dict[str, int] = {}
        for s in samples:
            counts[s] = counts.get(s, 0) + 1
        return counts

    # ── Full statevector reconstruction (small n only) ────────────────────

    def get_statevector(self) -> np.ndarray:
        """
        Reconstruct the full 2^n statevector. ONLY for n ≤ 24 (warns above 20).
        O(2^n · chi^2 · n) time and O(2^n) memory.
        """
        if self.n > 24:
            raise ValueError(f"Statevector reconstruction for n={self.n} > 24 not supported.")
        if self.n > 20:
            logger.warning("Reconstructing statevector for n=%d — this may be slow.", self.n)

        A = self.sites[0]  # (1, 2, chi_1)
        state = A[0, :, :]  # (2, chi_1)

        for i in range(1, self.n):
            B = self.sites[i]  # (chi_i, 2, chi_{i+1})
            state = np.tensordot(state, B, axes=([1], [0]))
            s = state.shape
            state = state.reshape(s[0] * s[1], s[2])

        sv = state[:, 0].copy()  # (2^n,) with site-0 as MSB

        if self.norm_log != 0.0:
            sv *= math.exp(self.norm_log)

        sv = sv.reshape([2] * self.n).transpose(
            list(range(self.n - 1, -1, -1))
        ).reshape(2 ** self.n)

        return sv

    # ── Entanglement entropy ─────────────────────────────────────────────────

    def entanglement_entropy(self, bond: int) -> float:
        """
        Compute von Neumann entropy S = -Σ_i λ_i² log(λ_i²) at the bond
        between sites `bond` and `bond+1` (0-indexed, 0 ≤ bond < n-1).
        Requires the MPS to be in mixed-canonical form at this bond.
        """
        if bond < 0 or bond >= self.n - 1:
            raise ValueError(f"bond {bond} out of range [0, {self.n-2}]")
        A = self.sites[0][0, :, :]
        for i in range(1, bond + 1):
            B = self.sites[i]
            A = np.tensordot(A, B, axes=([1], [0]))
            A = A.reshape(-1, A.shape[-1])

        _, S, _ = la.svd(A, full_matrices=False)
        S = S[S > 1e-15]
        S_sq = S**2
        S_sq /= S_sq.sum()
        return -float(np.sum(S_sq * np.log(np.maximum(S_sq, 1e-300))))

    def bond_spectrum(self, bond: int) -> np.ndarray:
        """Return singular values at bond `bond` (left-to-right sweep)."""
        mps = self.copy()
        for i in range(bond + 1):
            mps.left_normalize_site(i)
        A = mps.sites[bond]
        chi_l, d, chi_r = A.shape
        M = A.reshape(chi_l * d, chi_r)
        _, S, _ = la.svd(M, full_matrices=False)
        return S


# ─────────────────────────────────────────────────────────────────────────────
#  Gate-name classification (used by the circuit compiler)
# ─────────────────────────────────────────────────────────────────────────────

_SINGLE_QUBIT_GATES = frozenset({
    'id','h','x','y','z','s','sdg','t','tdg','sx','sxdg',
    'rx','ry','rz','p','phase','u1','u2','u3','u',
})

_TWO_QUBIT_GATES = frozenset({
    'cx','cnot','cy','cz','ch','cp','cphase',
    'crx','cry','crz','csx','swap','iswap','dcx',
    'ecr','rxx','ryy','rzz',
})
