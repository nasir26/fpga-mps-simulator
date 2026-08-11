#!/usr/bin/env python3
"""
surface_code_qec.py
====================
Distance-3 rotated surface-code quantum-memory experiment, run on the
28-qubit complex128 FPGA statevector simulator (fpga_simulator_v07.py /
opt_kernel.cpp on the Alveo U55C), implementing the stabilizer /
syndrome-extraction protocol described in:

  [12]  Dennis, Kitaev, Landahl, Preskill, "Topological quantum memory",
        arXiv:quant-ph/0110143 (12.pdf). Defines the surface-code check
        operators (their Eq. 4-5 / Fig. 1), the error model in which
        qubit bit-flip errors, qubit phase-flip errors *and syndrome-bit
        errors* all occur with the same probability p (their abstract /
        Sec. II), and the "quantum memory" experiment: encode, let errors
        accumulate over repeated syndrome rounds, decode, and check
        whether the logical value survived.

  [113] Fowler, Mariantoni, Martinis, Cleland, "Surface codes: Towards
        practical large-scale quantum computation", arXiv:1208.0928
        (113.pdf). Gives the concrete measure-Z / measure-X ancilla
        circuits (their Fig. 1b/c: CNOT(data->ancilla) for a Z-stabilizer;
        H, CNOT(ancilla->data)x4, H for an X-stabilizer) used here
        verbatim as the syndrome-extraction circuit.

Lattice
-------
9 data qubits (0-8) on a 3x3 grid, 8 ancilla qubits (9-16): 4 measure-X
(9-12) and 4 measure-Z (13-16). 17 qubits total, well inside the
simulator's 28-qubit / 2^28-amplitude HBM ceiling (MAX_QUBITS_FPGA in
fpga_simulator_v07.py).

    data grid (x,y), index = y*3+x:
        q0(0,0) q1(1,0) q2(2,0)
        q3(0,1) q4(1,1) q5(2,1)
        q6(0,2) q7(1,2) q8(2,2)

The 8 stabilizer generators and the two logical operators below were
*derived* (not copied from a figure) from first principles: place a
candidate ancilla at every half-integer position between the data grid
and one ring of padding around it, keep only those touching >=2 data
qubits, split them into two checkerboard colors by (x+y) mod 2, and at
each boundary keep only the color that gives that boundary a consistent
type (X-type ancillas truncated on top/bottom -> "X boundary" carries
logical X terminations; Z-type ancillas truncated on left/right ->
"Z boundary"). The resulting operators were verified programmatically in
`self_test()` below (weight-4/weight-2 stabilizers all mutually commute,
X-bar and Z-bar each commute with every stabilizer and anticommute with
each other) using qiskit.quantum_info as an independent check.

    X-stabilizers (detect Z errors):           Z-stabilizers (detect X errors):
      XS0 = X0 X1            (top bdry)          ZS0 = Z0 Z1 Z3 Z4   (interior)
      XS1 = X1 X2 X4 X5      (interior)          ZS1 = Z4 Z5 Z7 Z8   (interior)
      XS2 = X3 X4 X6 X7      (interior)          ZS2 = Z3 Z6         (left bdry)
      XS3 = X7 X8            (bottom bdry)       ZS3 = Z2 Z5         (right bdry)

    Logical X-bar = X0 X3 X6 (left column)   Logical Z-bar = Z0 Z1 Z2 (top row)

Hardware/host split
--------------------
fpga_simulator_v07's run_circuit() has no mid-circuit classical feedback:
`reset` deterministically collapses to |0> and measurement is Born-rule
shot-sampling of the *final* statevector, not a mid-circuit branch. So
each syndrome-extraction round is dispatched to the FPGA as one pure
unitary circuit (synthetic data errors + the measure-X/measure-Z CNOT
circuits of [113]); the projective measurement of the 8 ancillas (which
a real device would do in hardware) is performed on the host by sampling
the Born distribution of the *exact* statevector the FPGA returns, then
projecting/renormalizing -- this is the same measurement math a real
QPU's classical control system would apply to its own measurement
records, just executed on the host between kernel launches. The next
round's ancilla reset is then issued as literal 'x' gates on the FPGA
(so the reset itself still runs on the accelerator), and only the
classical MWPM decoding (which [12] Sec. II explicitly assigns to fast
classical processing, not the quantum device) runs on the host.

Usage
-----
    source /opt/xilinx/xrt/setup.sh   # if not already in your shell env
    python3 surface_code_qec.py --selftest
    python3 surface_code_qec.py --rounds 4 --trials 40 \
        --p 0.01 0.03 0.05 0.08 0.12 --logical-basis Z
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Bootstrap pyxrt's PYTHONPATH entry if it isn't already on sys.path (mirrors
# `source /opt/xilinx/xrt/setup.sh`'s PYTHONPATH export; LD_LIBRARY_PATH for
# the compiled libs is already exported by this project's .venv/bin/activate).
# ---------------------------------------------------------------------------
_XRT_PY = "/opt/xilinx/xrt/python"
if os.path.isdir(_XRT_PY) and _XRT_PY not in sys.path:
    sys.path.insert(0, _XRT_PY)

from fpga_simulator_v07 import QuantumFPGACircuit, MAX_QUBITS_FPGA  # noqa: E402

# =============================================================================
#  1. Lattice: data/ancilla qubit map, stabilizers, logical operators
# =============================================================================
NUM_DATA = 9
DATA = list(range(NUM_DATA))                     # 0..8
GRID = {q: (q % 3, q // 3) for q in DATA}         # index -> (x, y)

# X-type and Z-type ancilla candidate positions, generated exactly as
# described in the module docstring (kept explicit+inline here, not
# re-derived at runtime, so the geometry is easy to audit by eye).
X_SUPPORTS: List[FrozenSet[int]] = [
    frozenset({0, 1}),          # XS0 - top boundary
    frozenset({1, 2, 4, 5}),    # XS1 - interior
    frozenset({3, 4, 6, 7}),    # XS2 - interior
    frozenset({7, 8}),          # XS3 - bottom boundary
]
Z_SUPPORTS: List[FrozenSet[int]] = [
    frozenset({0, 1, 3, 4}),    # ZS0 - interior
    frozenset({4, 5, 7, 8}),    # ZS1 - interior
    frozenset({3, 6}),          # ZS2 - left boundary
    frozenset({2, 5}),          # ZS3 - right boundary
]
LOGICAL_X = frozenset({0, 3, 6})   # left column
LOGICAL_Z = frozenset({0, 1, 2})   # top row

# Qubit index assignment: data 0-8, X-ancillas 9-12, Z-ancillas 13-16.
NUM_ANCILLA = len(X_SUPPORTS) + len(Z_SUPPORTS)
NUM_QUBITS = NUM_DATA + NUM_ANCILLA
assert NUM_QUBITS <= MAX_QUBITS_FPGA, "lattice must fit the 28-qubit simulator"
X_ANCILLA = [NUM_DATA + i for i in range(len(X_SUPPORTS))]                       # 9..12
Z_ANCILLA = [NUM_DATA + len(X_SUPPORTS) + i for i in range(len(Z_SUPPORTS))]     # 13..16


def _raster_order(support: FrozenSet[int]) -> List[int]:
    """Deterministic CNOT order for one stabilizer: top-to-bottom,
    left-to-right by grid position. NOTE: this is a simple, *correct*
    ordering (all orderings of commuting-target CNOTs give the same
    final parity), but it is not the hook-error-optimal NW/NE/SW/SE
    schedule discussed in [113]; our noise model injects errors between
    rounds rather than mid-CNOT, so hook-error propagation is not
    exercised by this experiment. See module docstring."""
    return sorted(support, key=lambda q: (GRID[q][1], GRID[q][0]))


X_ORDER = [_raster_order(s) for s in X_SUPPORTS]
Z_ORDER = [_raster_order(s) for s in Z_SUPPORTS]


# =============================================================================
#  2. Decode graph: BFS distances over the space-time stabilizer graph, plus
#     shortest-path reconstruction so a matched pair of detection events can
#     be turned back into the set of data qubits it corrects.
# =============================================================================
class DecodeGraph:
    """Space-time matching graph for one stabilizer type (X or Z), built
    fresh for a given number of rounds R. Nodes are (stab_idx, round).
    A single shared virtual boundary node 'B' is reachable with weight 1
    from every boundary-type stabilizer (weight-2 support) in every
    round -- the standard simplification used by small matching decoders
    (e.g. PyMatching's virtual boundary node)."""

    BOUNDARY = "B"

    def __init__(self, supports: List[FrozenSet[int]], rounds: int):
        self.supports = supports
        self.n = len(supports)
        self.rounds = rounds
        # spatial adjacency: stabilizers sharing exactly one data qubit
        self.spatial_edge_qubit: Dict[Tuple[int, int], int] = {}
        for i, j in itertools.combinations(range(self.n), 2):
            shared = supports[i] & supports[j]
            if len(shared) == 1:
                (q,) = shared
                self.spatial_edge_qubit[(i, j)] = q
                self.spatial_edge_qubit[(j, i)] = q

        # General MWPM-graph boundary rule: each data qubit belongs to at
        # most 2 same-type stabilizers by construction. A qubit owned by
        # 2 stabilizers is already represented by the spatial edge above.
        # A qubit owned by exactly 1 stabilizer has no same-round partner
        # to pair with and must connect directly to the virtual boundary
        # -- this applies to *any* stabilizer that owns such a qubit, not
        # just the weight-2 "boundary" ones: an interior weight-4
        # stabilizer can still own a corner qubit no neighbour shares.
        # (An earlier version restricted boundary edges to weight-2
        # stabilizers only; that under-connects the graph and lets a
        # single weight-1 physical error decode to a logical error,
        # violating the d=3 guarantee -- caught by self_test().)
        owners: Dict[int, List[int]] = {}
        for i, supp in enumerate(supports):
            for q in supp:
                owners.setdefault(q, []).append(i)
        self.boundary_edges: List[Tuple[int, int]] = []   # (stab_idx, qubit)
        for q, own in owners.items():
            if len(own) == 1:
                self.boundary_edges.append((own[0], q))
            elif len(own) > 2:
                raise AssertionError(f"qubit {q} owned by {len(own)} same-type stabilizers")

        self._build_graph_adjacency()

    def _node(self, i, r):
        return (i, r)

    def _build_graph_adjacency(self):
        adj: Dict = {}

        def add_edge(a, b, kind, qubit):
            adj.setdefault(a, []).append((b, kind, qubit))
            adj.setdefault(b, []).append((a, kind, qubit))

        for r in range(self.rounds):
            for (i, j), q in self.spatial_edge_qubit.items():
                if i < j:
                    add_edge(self._node(i, r), self._node(j, r), "spatial", q)
            for i, q in self.boundary_edges:
                add_edge(self._node(i, r), self.BOUNDARY, "boundary", q)
        for r in range(self.rounds - 1):
            for i in range(self.n):
                add_edge(self._node(i, r), self._node(i, r + 1), "temporal", None)

        self.adj = adj

    def bfs(self, source):
        """Unweighted shortest paths (all edges have weight 1) from
        `source`. Returns (dist, parent_edge) where parent_edge[node] =
        (previous_node, edge_kind, qubit_or_None)."""
        dist = {source: 0}
        parent = {}
        q = deque([source])
        while q:
            u = q.popleft()
            for v, kind, qubit in self.adj.get(u, []):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent[v] = (u, kind, qubit)
                    q.append(v)
        return dist, parent

    def path_correction_qubits(self, a, b) -> List[int]:
        """Data qubits to flip for the shortest path between two nodes
        (or a node and self.BOUNDARY). Temporal edges contribute no
        physical correction (they represent a syndrome *measurement*
        error, not a data-qubit error)."""
        _, parent = self.bfs(a)
        if b not in parent and b != a:
            raise ValueError(f"{b} unreachable from {a}")
        qubits = []
        cur = b
        while cur != a:
            prev, kind, qubit = parent[cur]
            if kind in ("spatial", "boundary"):
                qubits.append(qubit)
            cur = prev
        return qubits


# =============================================================================
#  3. Exact (brute-force) minimum-weight perfect matching -- fine for the
#     small number of simultaneous detection events this d=3 experiment
#     produces at realistic p. Falls back to greedy nearest-neighbour
#     pairing if an unusually large event count is ever seen.
# =============================================================================
def _min_weight_perfect_matching(nodes: List, dist_fn) -> List[Tuple]:
    n = len(nodes)
    if n == 0:
        return []
    if n % 2 != 0:
        raise ValueError("matching requires an even number of nodes")
    if n > 12:  # (n-1)!! blows up; practically never hit at reasonable p
        return _greedy_matching(nodes, dist_fn)

    best = {"cost": math.inf, "pairs": None}

    def rec(remaining, pairs, cost):
        if cost >= best["cost"]:
            return
        if not remaining:
            best["cost"], best["pairs"] = cost, list(pairs)
            return
        a = remaining[0]
        rest = remaining[1:]
        for k, b in enumerate(rest):
            w = dist_fn(a, b)
            pairs.append((a, b))
            rec(rest[:k] + rest[k + 1:], pairs, cost + w)
            pairs.pop()

    rec(nodes, [], 0)
    return best["pairs"]


def _greedy_matching(nodes, dist_fn):
    remaining = list(nodes)
    pairs = []
    while remaining:
        a = remaining.pop(0)
        best_j, best_w = None, math.inf
        for j, b in enumerate(remaining):
            w = dist_fn(a, b)
            if w < best_w:
                best_w, best_j = w, j
        pairs.append((a, remaining.pop(best_j)))
    return pairs


def decode(graph: DecodeGraph, syndrome_rounds: List[List[int]]) -> List[int]:
    """syndrome_rounds[r][i] in {0,1} is the *recorded* eigenvalue bit of
    stabilizer i in round r. Returns the list of data qubits the decoder
    recommends flipping."""
    R = len(syndrome_rounds)
    detections = []
    prev = [0] * graph.n
    for r in range(R):
        cur = syndrome_rounds[r]
        for i in range(graph.n):
            if cur[i] != prev[i]:
                detections.append((i, r))
        prev = cur

    if not detections:
        return []

    # augment with one boundary-copy per real detection event so every
    # event can independently be matched "to the boundary" for free
    dist_to_b = {}
    for i, r in detections:
        d, _ = graph.bfs((i, r))
        dist_to_b[(i, r)] = d.get(graph.BOUNDARY, math.inf)

    real = detections
    copies = [("B", k) for k in range(len(real))]
    all_nodes = real + copies

    dist_cache: Dict[Tuple, Dict] = {}

    def bfs_dist(a):
        if a not in dist_cache:
            d, _ = graph.bfs(a)
            dist_cache[a] = d
        return dist_cache[a]

    def dist_fn(a, b):
        a_is_copy = isinstance(a, tuple) and a[0] == "B"
        b_is_copy = isinstance(b, tuple) and b[0] == "B"
        if a_is_copy and b_is_copy:
            return 0.0
        if a_is_copy:
            return dist_to_b[b]
        if b_is_copy:
            return dist_to_b[a]
        d = bfs_dist(a)
        return d.get(b, math.inf)

    matching = _min_weight_perfect_matching(all_nodes, dist_fn)

    correction = set()
    for a, b in matching:
        a_is_copy = isinstance(a, tuple) and a[0] == "B"
        b_is_copy = isinstance(b, tuple) and b[0] == "B"
        if a_is_copy and b_is_copy:
            continue  # two boundary copies matched to each other: no-op
        if a_is_copy or b_is_copy:
            real_node = b if a_is_copy else a
            qubits = graph.path_correction_qubits(real_node, graph.BOUNDARY)
        else:
            qubits = graph.path_correction_qubits(a, b)
        for q in qubits:
            correction.symmetric_difference_update({q})
    return sorted(correction)


# =============================================================================
#  4. Circuit builders (gate tuples in fpga_simulator_v07's native format)
# =============================================================================
def error_gates(x_errors: List[int], z_errors: List[int]) -> List[tuple]:
    gates = [("x", [q], None) for q in x_errors]
    gates += [("z", [q], None) for q in z_errors]
    return gates


def syndrome_round_gates(reset_x: List[int], reset_z: List[int]) -> List[tuple]:
    """One full round: physically reset any ancilla that read '1' last
    round (literal X gates, run on the FPGA), then the [113] Fig.1b/c
    measure-Z / measure-X circuits."""
    gates: List[tuple] = []
    for a in reset_x:
        gates.append(("x", [a], None))
    for a in reset_z:
        gates.append(("x", [a], None))

    for supp, order, anc in zip(Z_SUPPORTS, Z_ORDER, Z_ANCILLA):
        for q in order:
            gates.append(("cx", [q, anc], None))

    for supp, order, anc in zip(X_SUPPORTS, X_ORDER, X_ANCILLA):
        gates.append(("h", [anc], None))
        for q in order:
            gates.append(("cx", [anc, q], None))
        gates.append(("h", [anc], None))

    return gates


# =============================================================================
#  5. Host-side measurement: sample + project + (optionally) flip bits back
# =============================================================================
def sample_and_project(sv: np.ndarray, nq: int, meas_qubits: List[int],
                        rng: np.random.Generator) -> Tuple[Dict[int, int], np.ndarray]:
    """Exact joint projective measurement of `meas_qubits` in the Z basis,
    sampled from |sv|^2 (identical math to what a real device's Born-rule
    outcome statistics would produce), returning the sampled bits and the
    projected+renormalized statevector."""
    n = sv.shape[0]
    idx = np.arange(n, dtype=np.int64)
    keys = np.zeros(n, dtype=np.int64)
    for k, q in enumerate(meas_qubits):
        keys |= (((idx >> q) & 1) << k)

    probs = sv.real ** 2 + sv.imag ** 2
    probs = probs / probs.sum()
    nkeys = 1 << len(meas_qubits)
    marg = np.zeros(nkeys)
    np.add.at(marg, keys, probs)
    marg_sum = marg.sum()
    marg = marg / marg_sum if marg_sum > 0 else np.full(nkeys, 1.0 / nkeys)

    outcome = int(rng.choice(nkeys, p=marg))
    mask = keys == outcome
    new_sv = np.where(mask, sv, 0.0)
    nrm = np.linalg.norm(new_sv)
    new_sv = new_sv / nrm

    bits = {q: (outcome >> k) & 1 for k, q in enumerate(meas_qubits)}
    return bits, new_sv


# =============================================================================
#  6. Experiment driver: run R noisy rounds of the code on the FPGA,
#     collect synthetic syndrome data, decode, score the logical outcome.
# =============================================================================
@dataclass
class TrialResult:
    x_error_qubits: List[int]
    z_error_qubits: List[int]
    x_correction: List[int]
    z_correction: List[int]
    true_logical_flip: int      # ground truth, from injected errors (classical Pauli frame)
    hw_logical_bit: int         # ground truth, sampled from the real FPGA statevector
    decoded_logical_flip: int   # what the decoder thinks happened
    success: bool


def run_trial(fpga: QuantumFPGACircuit, rounds: int, p: float, q_meas: float,
              logical_basis: str, rng: np.random.Generator,
              x_graph: DecodeGraph, z_graph: DecodeGraph) -> TrialResult:
    nq = NUM_QUBITS

    # --- round 0: let the kernel init |0..0>, optionally Hadamard the data
    #     qubits into |+>^9 for an X-bar-logical ('X' basis) experiment.
    prep = [("h", [q], None) for q in DATA] if logical_basis == "X" else []
    reset_x: List[int] = []
    reset_z: List[int] = []

    true_x_err = [0] * NUM_DATA   # cumulative classical Pauli frame (ground truth)
    true_z_err = [0] * NUM_DATA
    x_syndrome_rounds: List[List[int]] = []
    z_syndrome_rounds: List[List[int]] = []

    sv = None
    for r in range(rounds):
        x_err_q = [d for d in DATA if rng.random() < p]
        z_err_q = [d for d in DATA if rng.random() < p]
        for d in x_err_q:
            true_x_err[d] ^= 1
        for d in z_err_q:
            true_z_err[d] ^= 1

        gates = list(prep) if r == 0 else []
        gates += error_gates(x_err_q, z_err_q)
        gates += syndrome_round_gates(reset_x, reset_z)

        if r == 0:
            res = fpga.run_circuit(nq, gates, shots=0, validate=False,
                                    circuit_name=f"sc_round{r}")
        else:
            fpga._distribute_arbitrary_state(sv, nq)
            res = fpga.run_circuit(nq, gates, shots=0, validate=False,
                                    circuit_name=f"sc_round{r}", skip_init=True)
        sv = res["statevector"]

        bits, sv = sample_and_project(sv, nq, X_ANCILLA + Z_ANCILLA, rng)
        x_bits_true = [bits[a] for a in X_ANCILLA]
        z_bits_true = [bits[a] for a in Z_ANCILLA]
        reset_x = [a for a in X_ANCILLA if bits[a] == 1]
        reset_z = [a for a in Z_ANCILLA if bits[a] == 1]

        # synthetic (noisy) syndrome data seen by the decoder: perfect on
        # the final round (standard time-boundary closure), corrupted with
        # independent probability q_meas otherwise -- paper [12]'s model
        # applies the *same* p to qubit errors and syndrome-bit errors, so
        # by default q_meas == p (see main()).
        if r == rounds - 1:
            x_syndrome_rounds.append(x_bits_true)
            z_syndrome_rounds.append(z_bits_true)
        else:
            x_syndrome_rounds.append([b ^ (1 if rng.random() < q_meas else 0) for b in x_bits_true])
            z_syndrome_rounds.append([b ^ (1 if rng.random() < q_meas else 0) for b in z_bits_true])

    x_correction = decode(x_graph, x_syndrome_rounds)   # corrects Z errors
    z_correction = decode(z_graph, z_syndrome_rounds)   # corrects X errors

    # classical Pauli-frame ground truth for the logical bit under test
    if logical_basis == "Z":
        true_flip = sum(true_x_err[d] for d in LOGICAL_Z) % 2
        dec_flip = sum(1 for d in z_correction if d in LOGICAL_Z) % 2
    else:
        true_flip = sum(true_z_err[d] for d in LOGICAL_X) % 2
        dec_flip = sum(1 for d in x_correction if d in LOGICAL_X) % 2

    # independent hardware cross-check: sample the *true* final data-qubit
    # register from the exact FPGA statevector and read the logical bit
    # off it directly (see module docstring: the logical parity being
    # tracked commutes with every stabilizer, so this is deterministic).
    data_bits, _ = sample_and_project(sv, nq, DATA, rng)
    if logical_basis == "Z":
        hw_bit = sum(data_bits[d] for d in LOGICAL_Z) % 2
    else:
        hw_bit = sum(data_bits[d] for d in LOGICAL_X) % 2
    # expected hw_bit (before any correction) == true_flip when starting
    # from a deterministic +1 eigenstate of the tracked logical operator
    if hw_bit != true_flip:
        raise AssertionError(
            f"HW/classical Pauli-frame mismatch: hw_bit={hw_bit} true_flip={true_flip} "
            "(statevector kernel and host bookkeeping disagree)")

    success = (true_flip == dec_flip)
    return TrialResult(x_error_qubits=[d for d in DATA if true_x_err[d]],
                        z_error_qubits=[d for d in DATA if true_z_err[d]],
                        x_correction=x_correction, z_correction=z_correction,
                        true_logical_flip=true_flip, hw_logical_bit=hw_bit,
                        decoded_logical_flip=dec_flip, success=success)


# =============================================================================
#  7. Self-test: pure-software verification of the lattice/decoder (no HW)
# =============================================================================
def self_test(verbose: bool = True) -> bool:
    from qiskit.quantum_info import Pauli, Statevector

    def pauli_string(support: FrozenSet[int], letter: str) -> str:
        chars = ["I"] * NUM_DATA
        for q in support:
            chars[q] = letter
        return "".join(reversed(chars))  # qiskit little-endian

    ok = True

    def check(name, cond):
        nonlocal ok
        mark = "PASS" if cond else "FAIL"
        if verbose:
            print(f"  [{mark}] {name}")
        ok = ok and cond

    x_paulis = [Pauli(pauli_string(s, "X")) for s in X_SUPPORTS]
    z_paulis = [Pauli(pauli_string(s, "Z")) for s in Z_SUPPORTS]
    xbar = Pauli(pauli_string(LOGICAL_X, "X"))
    zbar = Pauli(pauli_string(LOGICAL_Z, "Z"))

    for i, a in enumerate(x_paulis):
        for j, b in enumerate(x_paulis):
            check(f"XS{i} commutes XS{j}", a.commutes(b))
    for i, a in enumerate(z_paulis):
        for j, b in enumerate(z_paulis):
            check(f"ZS{i} commutes ZS{j}", a.commutes(b))
    for i, a in enumerate(x_paulis):
        for j, b in enumerate(z_paulis):
            check(f"XS{i} commutes ZS{j}", a.commutes(b))
    for i, a in enumerate(x_paulis):
        check(f"Xbar commutes XS{i}", xbar.commutes(a))
        check(f"Zbar commutes XS{i}", zbar.commutes(a))
    for i, a in enumerate(z_paulis):
        check(f"Xbar commutes ZS{i}", xbar.commutes(a))
        check(f"Zbar commutes ZS{i}", zbar.commutes(a))
    check("Xbar anticommutes Zbar", not xbar.commutes(zbar))

    # code state: |0>^9 projected onto the +1 eigenspace of every
    # X-stabilizer via explicit projector application (pure linear algebra,
    # independent of the FPGA circuit), then verify every stabilizer and
    # Zbar read back out at the eigenvalues we expect.
    # build |0>^9 -> apply (I + XS_i)/norm projectors sequentially
    state = np.zeros(2 ** NUM_DATA, dtype=complex)
    state[0] = 1.0
    from qiskit.quantum_info import Operator
    ident = Operator(Pauli("I" * NUM_DATA)).data
    for xp in x_paulis:
        proj = 0.5 * (ident + Operator(xp).data)
        state = proj @ state
        nrm = np.linalg.norm(state)
        if nrm > 1e-12:
            state = state / nrm
    sv = Statevector(state)
    for i, zp in enumerate(z_paulis):
        ev = sv.expectation_value(zp).real
        check(f"code state: <ZS{i}> == +1", abs(ev - 1.0) < 1e-8)
    for i, xp in enumerate(x_paulis):
        ev = sv.expectation_value(xp).real
        check(f"code state: <XS{i}> == +1", abs(ev - 1.0) < 1e-8)
    ev = sv.expectation_value(zbar).real
    check("code state: <Zbar> == +1", abs(ev - 1.0) < 1e-8)

    # Decoder sanity, checked against the scientifically correct notion of
    # success ([12] Sec. III): a correction succeeds iff (true_error XOR
    # correction) has EVEN overlap with the logical operator's support --
    # i.e. it leaves the logical value unchanged -- not that it reproduces
    # the exact error location. Some qubits here are genuinely
    # syndrome-degenerate (a single Z-error on q2 or on q5 both trigger
    # XS1 alone, since both are owned solely by XS1): the decoder cannot,
    # and does not need to, tell them apart -- only preserve Xbar.
    graph = DecodeGraph(X_SUPPORTS, rounds=1)
    for q in DATA:
        cur = [0, 0, 0, 0]
        for i, s in enumerate(X_SUPPORTS):
            if q in s:
                cur[i] ^= 1
        correction = decode(graph, [cur])
        residual = {q} ^ set(correction)
        logical_ok = (len(residual & LOGICAL_X) % 2) == 0
        check(f"single Z-error on q{q}: decoder preserves logical Xbar "
              f"(correction={correction})", logical_ok)

    # a weight-2 error spanning the whole lattice column-wise (same as
    # Zbar's own support pattern under Z errors is not applicable here;
    # instead check two independent single-qubit errors decode
    # independently when far apart)
    graph2 = DecodeGraph(X_SUPPORTS, rounds=1)
    cur = [0, 0, 0, 0]
    for q in (0, 8):
        for i, s in enumerate(X_SUPPORTS):
            if q in s:
                cur[i] ^= 1
    correction = decode(graph2, [cur])
    check("two far-apart single Z-errors (q0,q8) decode to {q0,q8}",
          set(correction) == {0, 8})

    # symmetric check on the Z-graph: single X-errors threaten Zbar
    graph3 = DecodeGraph(Z_SUPPORTS, rounds=1)
    for q in DATA:
        cur = [0, 0, 0, 0]
        for i, s in enumerate(Z_SUPPORTS):
            if q in s:
                cur[i] ^= 1
        correction = decode(graph3, [cur])
        residual = {q} ^ set(correction)
        logical_ok = (len(residual & LOGICAL_Z) % 2) == 0
        check(f"single X-error on q{q}: decoder preserves logical Zbar "
              f"(correction={correction})", logical_ok)

    if verbose:
        print(f"\nself_test: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return ok


# =============================================================================
#  8. Main: sweep p on real FPGA hardware, report + plot logical error rate
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true",
                     help="run pure-software lattice/decoder verification and exit")
    ap.add_argument("--xclbin", default="quantum_fpga_kernel_complex128.xclbin")
    ap.add_argument("--rounds", type=int, default=4, help="syndrome-extraction rounds per trial")
    ap.add_argument("--trials", type=int, default=40, help="trials per physical error rate p")
    ap.add_argument("--p", type=float, nargs="+", default=[0.01, 0.03, 0.05, 0.08, 0.12],
                     help="physical error rate(s) to sweep")
    ap.add_argument("--q-meas", type=float, default=None,
                     help="syndrome-bit readout error rate (default: equal to p, per [12])")
    ap.add_argument("--logical-basis", choices=["Z", "X"], default="Z",
                     help="track logical Z-bar (bit-flip memory) or X-bar (phase-flip memory)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-prefix", default="surface_code_d3")
    args = ap.parse_args()

    if args.selftest:
        ok = self_test()
        sys.exit(0 if ok else 1)

    print(f"[surface_code_qec] d=3 rotated surface code: {NUM_DATA} data + "
          f"{NUM_ANCILLA} ancilla = {NUM_QUBITS} qubits (of {MAX_QUBITS_FPGA} available)")
    print("[surface_code_qec] running self_test() before touching hardware...")
    if not self_test(verbose=False):
        print("[FATAL] self_test failed -- refusing to run on hardware.")
        sys.exit(1)
    print("[surface_code_qec] self_test PASS")

    fpga = QuantumFPGACircuit(args.xclbin)
    x_graph = DecodeGraph(X_SUPPORTS, rounds=args.rounds)
    z_graph = DecodeGraph(Z_SUPPORTS, rounds=args.rounds)
    rng = np.random.default_rng(args.seed)

    rows = []
    for p in args.p:
        q_meas = args.q_meas if args.q_meas is not None else p
        n_fail = 0
        t0 = time.perf_counter()
        for t in range(args.trials):
            r = run_trial(fpga, args.rounds, p, q_meas, args.logical_basis, rng, x_graph, z_graph)
            if not r.success:
                n_fail += 1
        dt = time.perf_counter() - t0
        rate = n_fail / args.trials
        rows.append((p, args.trials, n_fail, rate))
        print(f"  p={p:.4f}  trials={args.trials:4d}  logical_fail={n_fail:4d}  "
              f"logical_error_rate={rate:.4f}   ({dt:.1f}s, {dt/args.trials*1e3:.1f} ms/trial)")

    csv_path = f"{args.out_prefix}_results.csv"
    with open(csv_path, "w") as f:
        f.write("p,trials,logical_fail,logical_error_rate\n")
        for p, n, nf, rate in rows:
            f.write(f"{p},{n},{nf},{rate}\n")
    print(f"[surface_code_qec] wrote {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ps = [r[0] for r in rows]
        rates = [r[3] for r in rows]
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(ps, ps, "k--", label="physical error rate p (uncorrected)")
        ax.plot(ps, rates, "o-", color="tab:blue",
                label=f"decoded logical error rate (d=3, R={args.rounds})")
        ax.set_xlabel("physical error rate p")
        ax.set_ylabel("logical error rate")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Distance-3 rotated surface code quantum memory\n"
                      f"({args.logical_basis}-bar, synthetic syndrome data, {NUM_QUBITS}-qubit FPGA statevector sim)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        png_path = f"{args.out_prefix}_logical_error_rate.png"
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        print(f"[surface_code_qec] wrote {png_path}")
    except Exception as e:
        print(f"[surface_code_qec] plotting skipped: {e}")


if __name__ == "__main__":
    main()
