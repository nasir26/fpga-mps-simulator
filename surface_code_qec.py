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

`build_lattice(dx, dy)` below generalizes this derivation to an arbitrary
dx-by-dy rectangular data grid (future-work item (i), "scaling within the
28-qubit budget to larger code patches"): it reproduces the hand-derived
3x3 lattice above exactly (verified in self_test()) and is also used to
build a larger 3x4 patch (12 data + 11 ancilla = 23 of 28 qubits) that
is independently verified with the same algebraic/decoding self-test.
Two full independent d=3 patches (17+17=34 qubits) do not fit the
28-qubit budget, which is why the demonstrated scale-up is a single
larger rectangular patch rather than a second logical qubit -- see the
module-level LARGER_LATTICE and --dx/--dy CLI flags.

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

Decoders (future-work item (iii))
----------------------------------
Two interchangeable decoder backends operate on the same DecodeGraph:
  - "bruteforce": the original exact brute-force MWPM solver (fine for
    the handful of simultaneous detection events a d=3/d=4 patch
    produces at realistic p).
  - "pymatching": the same space-time graph, converted once per sweep
    into a pymatching.Matching object (weight-1 spatial/boundary edges
    carry the corrected data qubit as a fault_id; weight-1 temporal
    edges carry none), decoded with PyMatching's blossom-based MWPM.
    Selected via --decoder pymatching. self_test() cross-checks that
    both backends agree on every exhaustive single-qubit-error case.

Noise model (future-work item (iv))
-------------------------------------
--noise-model independent (default) reproduces [12]'s model exactly:
independent X and Z channels at probability p (unchanged from the
published version of this experiment). --noise-model depolarizing
applies a genuine single-qubit depolarizing channel instead: with total
probability p, one of {X, Y, Z} is applied with equal probability p/3
each (Y realized as simultaneous X+Z Pauli-frame flips, exactly as
Y = iXZ acts on stabilizer eigenvalues). --leakage-prob adds, independent
of the Pauli channel, a simplified leakage proxy: with the given
probability a data qubit is hit with `reset` (the kernel's only
non-unitary primitive) followed by a 50% chance of `x`, i.e. it is
projected out of whatever state it was in and re-emerges in a uniformly
random computational basis state. This is a deliberately simple
stand-in for leakage (the kernel has no third level to leak into, so
"leakage" here means "this round destroys the qubit's frame-trackable
Pauli history"), not a physical leakage channel -- see run_trial() for
how this breaks the classical Pauli-frame ground truth and why the
trial then falls back to the FPGA's own statevector readout as ground
truth instead.

Stim cross-validation (future-work item (v))
-----------------------------------------------
--stim-crossval replays, on a stim.TableauSimulator, the *exact same*
gate stream (errors + ancilla resets + measure-X/measure-Z circuits)
dispatched to the FPGA each round, then compares Stim's per-ancilla
measurement outcome against the FPGA's Born-rule-sampled outcome for
that round. Because every gate in this experiment is Clifford and every
injected error is a plain Pauli operator, the ancilla outcome is not
actually random -- it is the deterministic eigenvalue of a stabilizer
generator given the accumulated error -- so this is an exact, bit-level
cross-check (not a statistical one), run on real hardware trials in
addition to the existing classical-Pauli-frame cross-check. See
StimCrossValidator below.

Usage
-----
    source /opt/xilinx/xrt/setup.sh   # if not already in your shell env
    python3 surface_code_qec.py --selftest
    python3 surface_code_qec.py --rounds 4 --trials 40 \
        --p 0.01 0.03 0.05 0.08 0.12 --logical-basis Z
    python3 surface_code_qec.py --rounds 4 --trials 20 --p 0.03 \
        --decoder pymatching --noise-model depolarizing \
        --leakage-prob 0.01 --stim-crossval
    python3 surface_code_qec.py --rounds 4 --trials 20 --p 0.03 \
        --dx 3 --dy 4   # larger patch, 23 of 28 qubits
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
@dataclass
class Lattice:
    """One rotated-surface-code patch: dx-by-dy data grid, its derived
    stabilizer generators, logical operators, and the resulting qubit
    index assignment (data first, then X-ancillas, then Z-ancillas)."""
    dx: int
    dy: int
    num_data: int
    grid: Dict[int, Tuple[int, int]]
    x_supports: List[FrozenSet[int]]
    z_supports: List[FrozenSet[int]]
    logical_x: FrozenSet[int]
    logical_z: FrozenSet[int]

    def __post_init__(self):
        self.data = list(range(self.num_data))
        self.num_ancilla = len(self.x_supports) + len(self.z_supports)
        self.num_qubits = self.num_data + self.num_ancilla
        assert self.num_qubits <= MAX_QUBITS_FPGA, \
            f"lattice ({self.dx}x{self.dy}, {self.num_qubits} qubits) exceeds the {MAX_QUBITS_FPGA}-qubit budget"
        self.x_ancilla = [self.num_data + i for i in range(len(self.x_supports))]
        self.z_ancilla = [self.num_data + len(self.x_supports) + i
                           for i in range(len(self.z_supports))]
        self.x_order = [_raster_order(s, self.grid) for s in self.x_supports]
        self.z_order = [_raster_order(s, self.grid) for s in self.z_supports]


def _raster_order(support: FrozenSet[int], grid: Dict[int, Tuple[int, int]]) -> List[int]:
    """Deterministic CNOT order for one stabilizer: top-to-bottom,
    left-to-right by grid position. NOTE: this is a simple, *correct*
    ordering (all orderings of commuting-target CNOTs give the same
    final parity), but it is not the hook-error-optimal NW/NE/SW/SE
    schedule discussed in [113]; our noise model injects errors between
    rounds rather than mid-CNOT, so hook-error propagation is not
    exercised by this experiment. See module docstring."""
    return sorted(support, key=lambda q: (grid[q][1], grid[q][0]))


def build_lattice(dx: int, dy: int) -> Lattice:
    """General derivation of a dx-by-dy rotated-surface-code-style patch,
    generalizing the hand-derivation in the module docstring: place a
    candidate ancilla at every half-integer position between the data
    grid and one ring of padding around it, keep only candidates
    touching >=2 data qubits, and at each of the four physical edges keep
    only the checkerboard color that gives that edge a consistent
    boundary type (X-type truncated top/bottom, Z-type truncated
    left/right). Verified in self_test() to reproduce the hardcoded 3x3
    lattice exactly (as sets of stabilizer supports) and, for a second,
    larger instance, to satisfy the same commutation/logical/decoding
    invariants -- future-work item (i)."""
    num_data = dx * dy
    grid = {q: (q % dx, q // dx) for q in range(num_data)}
    coord_to_q = {v: k for k, v in grid.items()}

    x_supports: List[FrozenSet[int]] = []
    z_supports: List[FrozenSet[int]] = []
    for x in range(-1, dx):
        for y in range(-1, dy):
            corners = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)]
            support = frozenset(coord_to_q[c] for c in corners if c in coord_to_q)
            if len(support) < 2:
                continue  # true lattice corner: touches only 1 data qubit
            is_x_type = (x + y) % 2 == 1
            top_bottom_pad = y in (-1, dy - 1) and len(support) == 2
            left_right_pad = x in (-1, dx - 1) and len(support) == 2
            if top_bottom_pad and not is_x_type:
                continue
            if left_right_pad and is_x_type:
                continue
            (x_supports if is_x_type else z_supports).append(support)

    logical_x = frozenset(coord_to_q[(0, y)] for y in range(dy))   # left column
    logical_z = frozenset(coord_to_q[(x, 0)] for x in range(dx))   # top row

    return Lattice(dx=dx, dy=dy, num_data=num_data, grid=grid,
                    x_supports=x_supports, z_supports=z_supports,
                    logical_x=logical_x, logical_z=logical_z)


# Default d=3 patch, kept as the original hand-derived/hardcoded values
# (byte-identical stabilizer ordering to every prior run of this
# experiment and to the paper's Table 1) rather than routed through
# build_lattice(), so existing qubit-index/ordering assumptions and
# published results are unaffected.
DEFAULT_LATTICE = Lattice(
    dx=3, dy=3, num_data=9, grid={q: (q % 3, q // 3) for q in range(9)},
    x_supports=[
        frozenset({0, 1}),          # XS0 - top boundary
        frozenset({1, 2, 4, 5}),    # XS1 - interior
        frozenset({3, 4, 6, 7}),    # XS2 - interior
        frozenset({7, 8}),          # XS3 - bottom boundary
    ],
    z_supports=[
        frozenset({0, 1, 3, 4}),    # ZS0 - interior
        frozenset({4, 5, 7, 8}),    # ZS1 - interior
        frozenset({3, 6}),          # ZS2 - left boundary
        frozenset({2, 5}),          # ZS3 - right boundary
    ],
    logical_x=frozenset({0, 3, 6}),   # left column
    logical_z=frozenset({0, 1, 2}),   # top row
)

# Larger patch demonstrating future-work item (i) ("scaling within the
# 28-qubit budget to larger code patches"): 12 data + 11 ancilla = 23 of
# 28 qubits (two independent d=3 patches would need 34 qubits and do not
# fit). Distance min(dx,dy)=3 against one error type, 4 against the
# other. Verified by the same self-test rigor as DEFAULT_LATTICE.
LARGER_LATTICE = build_lattice(3, 4)


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
        self._pymatching_cache = None  # lazily built by build_pymatching_matching()

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
    recommends flipping. Exact brute-force MWPM backend (default)."""
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
#  3b. PyMatching backend (future-work item (iii): "replacing the
#      brute-force matching solver with PyMatching or a Union-Find
#      decoder to support larger code distances"). Builds the identical
#      space-time graph as DecodeGraph into a pymatching.Matching once
#      per sweep, then decodes each trial's detection-event vector with
#      PyMatching's blossom-based MWPM.
# =============================================================================
def build_pymatching_matching(graph: DecodeGraph):
    """Convert a DecodeGraph into a pymatching.Matching with an identical
    edge set: spatial/boundary edges carry the data qubit they'd correct
    as a fault_id (so decode()'s correction vector is indexed by data
    qubit directly), temporal edges carry no fault_id (a measurement
    error contributes no physical-qubit correction, matching decode()'s
    brute-force semantics)."""
    import pymatching

    def node_id(i, r):
        return r * graph.n + i

    # merge_strategy="smallest-weight": a stabilizer can own >1 boundary
    # qubit with no same-round neighbour (the "syndrome-degenerate" case
    # documented in the module docstring/paper, e.g. XS1 owning both q2
    # and q5 in the default lattice) -- pymatching represents the
    # boundary as a single virtual node, so a second same-weight
    # boundary edge from the same stabilizer node is a parallel edge and
    # is kept-as-first rather than merged, matching the brute-force
    # graph's own (equally arbitrary) first-discovered BFS tie-break.
    m = pymatching.Matching()
    for r in range(graph.rounds):
        for (i, j), q in graph.spatial_edge_qubit.items():
            if i < j:
                m.add_edge(node_id(i, r), node_id(j, r), fault_ids={q}, weight=1,
                           merge_strategy="smallest-weight")
        for i, q in graph.boundary_edges:
            m.add_boundary_edge(node_id(i, r), fault_ids={q}, weight=1,
                                 merge_strategy="smallest-weight")
    for r in range(graph.rounds - 1):
        for i in range(graph.n):
            m.add_edge(node_id(i, r), node_id(i, r + 1), weight=1,
                       merge_strategy="smallest-weight")
    return m


def decode_pymatching(matching, graph: DecodeGraph,
                       syndrome_rounds: List[List[int]]) -> List[int]:
    """Same contract as decode(), backed by PyMatching instead of the
    brute-force solver. `matching` is a pymatching.Matching built once
    by build_pymatching_matching(graph) and reused across trials."""
    R = len(syndrome_rounds)
    detections = np.zeros(graph.n * R, dtype=np.uint8)
    prev = [0] * graph.n
    for r in range(R):
        cur = syndrome_rounds[r]
        for i in range(graph.n):
            if cur[i] != prev[i]:
                detections[r * graph.n + i] = 1
        prev = cur
    if not detections.any():
        return []
    correction = matching.decode(detections)
    return [q for q, bit in enumerate(correction) if bit]


# =============================================================================
#  4. Circuit builders (gate tuples in fpga_simulator_v07's native format)
# =============================================================================
def error_gates(x_errors: List[int], z_errors: List[int]) -> List[tuple]:
    gates = [("x", [q], None) for q in x_errors]
    gates += [("z", [q], None) for q in z_errors]
    return gates


def sample_data_errors(rng: np.random.Generator, data_qubits: List[int], p: float,
                        noise_model: str, leak_p: float) -> Tuple[List[int], List[int], List[int]]:
    """One round's synthetic error draw for every data qubit.

    noise_model == "independent" (default): reproduces [12]'s model
    exactly -- independent X and Z channels, each at probability p
    (unchanged from the original published version of this experiment).

    noise_model == "depolarizing": a genuine single-qubit depolarizing
    channel -- with total probability p, exactly one of {X, Y, Z} is
    applied, each with probability p/3. Y is realized as simultaneous
    X+Z Pauli-frame flips (Y = iXZ acts on stabilizer eigenvalues
    exactly like an X error and a Z error occurring together).

    leak_p (independent of the Pauli channel above): probability that a
    data qubit "leaks" this round -- see module docstring and
    leakage_gates() below. Returned separately since it is not a Pauli
    operator and must be handled differently by the caller."""
    x_err: List[int] = []
    z_err: List[int] = []
    leaked: List[int] = []
    for d in data_qubits:
        if noise_model == "depolarizing":
            if rng.random() < p:
                kind = rng.integers(0, 3)  # 0=X, 1=Y, 2=Z, each p/3
                if kind == 0:
                    x_err.append(d)
                elif kind == 1:
                    x_err.append(d)
                    z_err.append(d)
                else:
                    z_err.append(d)
        elif noise_model == "independent":
            if rng.random() < p:
                x_err.append(d)
            if rng.random() < p:
                z_err.append(d)
        else:
            raise ValueError(f"unknown noise_model {noise_model!r}")
        if leak_p > 0.0 and rng.random() < leak_p:
            leaked.append(d)
    return x_err, z_err, leaked


def leakage_gates(leaked: List[int], rng: np.random.Generator) -> List[tuple]:
    """Simplified leakage proxy (future-work item (iv)): `reset` (the
    kernel's only non-unitary primitive) projects the qubit to |0>,
    then a 50% coin flip re-randomizes it to |1>. This is NOT a physical
    leakage channel -- the kernel has no third level -- it is a
    documented stand-in for "this qubit's prior state, and any Pauli
    frame tracked for it, is destroyed and replaced by a uniformly
    random computational-basis outcome this round." See run_trial() for
    how a trial that hits this path falls back to the FPGA's own
    statevector readout as ground truth instead of the classical
    Pauli-frame accumulator."""
    gates: List[tuple] = []
    for d in leaked:
        gates.append(("reset", [d], None))
        if rng.random() < 0.5:
            gates.append(("x", [d], None))
    return gates


def syndrome_round_gates(lattice: Lattice, reset_x: List[int], reset_z: List[int]) -> List[tuple]:
    """One full round: physically reset any ancilla that read '1' last
    round (literal X gates, run on the FPGA), then the [113] Fig.1b/c
    measure-Z / measure-X circuits."""
    gates: List[tuple] = []
    for a in reset_x:
        gates.append(("x", [a], None))
    for a in reset_z:
        gates.append(("x", [a], None))

    for order, anc in zip(lattice.z_order, lattice.z_ancilla):
        for q in order:
            gates.append(("cx", [q, anc], None))

    for order, anc in zip(lattice.x_order, lattice.x_ancilla):
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
#  5b. Stim cross-validation (future-work item (v))
# =============================================================================
class StimCrossValidator:
    """Cross-check of one syndrome round's ancilla readout against an
    independent Stim stabilizer-circuit simulation of the *same* gate
    sequence dispatched to the FPGA that round.

    Every gate this experiment dispatches -- Pauli error injection,
    ancilla reset, and the CNOT/H syndrome-extraction circuits -- is
    Clifford. For a Z-type ancilla (measuring a Z-stabilizer of a data
    register that starts in |0>^n, a Z eigenstate) the outcome is fully
    deterministic. But an X-type ancilla's *first* measurement is
    genuinely random -- |0>^n is not an eigenstate of the X-stabilizers,
    so the very first round's X-syndrome outcome is a real Born-rule
    coin flip that establishes which code-space branch the experiment is
    in, exactly as it would on real hardware. Two independently-seeded
    simulators (the FPGA's host-side Born-rule sampler and Stim's own
    RNG) are not expected to land on the same branch by chance, so
    naively comparing raw measured bits produces spurious mismatches
    that are not bugs.

    Instead, `check()` *postselects* Stim onto the FPGA's already-sampled
    outcome for every ancilla, every round (stim.TableauSimulator.
    postselect_z forces that outcome and keeps Stim locked to the same
    branch the real experiment realized). This works uniformly whether
    the outcome was "genuinely random" (round 0, X-ancillas) or fully
    determined by the accumulated error (every other case): postselection
    only raises if the FPGA's reported outcome is *impossible* under
    Stim's independent tracking of the same gate sequence -- i.e. iff the
    two implementations disagree about the circuit's stabilizer physics,
    which is the actual bug this cross-check is designed to catch."""

    def __init__(self):
        import stim
        self.sim = stim.TableauSimulator()
        self.checked = 0
        self.mismatches = 0

    def replay(self, gates: List[tuple]) -> None:
        for name, qs, _ in gates:
            if name == "x":
                self.sim.x(*qs)
            elif name == "z":
                self.sim.z(*qs)
            elif name == "h":
                self.sim.h(*qs)
            elif name == "cx":
                self.sim.cx(*qs)
            elif name == "reset":
                self.sim.reset(*qs)
            else:
                raise ValueError(f"stim cross-val: unsupported gate {name!r}")

    def check(self, ancillas: List[int], fpga_bits: Dict[int, int]) -> bool:
        ok = True
        for a in ancillas:
            self.checked += 1
            try:
                self.sim.postselect_z(a, desired_value=bool(fpga_bits[a]))
            except ValueError:
                ok = False
                self.mismatches += 1
        return ok


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
    true_logical_flip: int      # ground truth (Pauli frame, or hw_bit if leakage occurred)
    hw_logical_bit: int         # ground truth, sampled from the real FPGA statevector
    decoded_logical_flip: int   # what the decoder thinks happened
    success: bool
    leaked: bool = False        # whether any leakage event occurred this trial
    stim_mismatches: int = 0    # ancilla bits where stim disagreed with the FPGA, this trial


def run_trial(fpga: QuantumFPGACircuit, lattice: Lattice, rounds: int, p: float, q_meas: float,
              logical_basis: str, rng: np.random.Generator,
              x_graph: DecodeGraph, z_graph: DecodeGraph,
              decoder: str = "bruteforce",
              x_matching=None, z_matching=None,
              noise_model: str = "independent", leak_p: float = 0.0,
              stim_validator: Optional[StimCrossValidator] = None) -> TrialResult:
    nq = lattice.num_qubits
    DATA = lattice.data
    X_ANCILLA, Z_ANCILLA = lattice.x_ancilla, lattice.z_ancilla
    LOGICAL_X, LOGICAL_Z = lattice.logical_x, lattice.logical_z

    # --- round 0: let the kernel init |0..0>, optionally Hadamard the data
    #     qubits into |+>^9 for an X-bar-logical ('X' basis) experiment.
    prep = [("h", [q], None) for q in DATA] if logical_basis == "X" else []
    reset_x: List[int] = []
    reset_z: List[int] = []

    true_x_err = [0] * lattice.num_data   # cumulative classical Pauli frame (ground truth)
    true_z_err = [0] * lattice.num_data
    x_syndrome_rounds: List[List[int]] = []
    z_syndrome_rounds: List[List[int]] = []
    any_leak = False
    stim_ok = True

    sv = None
    for r in range(rounds):
        x_err_q, z_err_q, leaked_q = sample_data_errors(rng, DATA, p, noise_model, leak_p)
        for d in x_err_q:
            true_x_err[d] ^= 1
        for d in z_err_q:
            true_z_err[d] ^= 1
        if leaked_q:
            any_leak = True

        gates = list(prep) if r == 0 else []
        gates += error_gates(x_err_q, z_err_q)
        gates += leakage_gates(leaked_q, rng)
        gates += syndrome_round_gates(lattice, reset_x, reset_z)

        if r == 0:
            res = fpga.run_circuit(nq, gates, shots=0, validate=False,
                                    circuit_name=f"sc_round{r}")
        else:
            fpga._distribute_arbitrary_state(sv, nq)
            res = fpga.run_circuit(nq, gates, shots=0, validate=False,
                                    circuit_name=f"sc_round{r}", skip_init=True)
        sv = res["statevector"]

        bits, sv = sample_and_project(sv, nq, X_ANCILLA + Z_ANCILLA, rng)

        if stim_validator is not None:
            stim_validator.replay(gates)
            if not stim_validator.check(X_ANCILLA + Z_ANCILLA, bits):
                stim_ok = False

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

    if decoder == "bruteforce":
        x_correction = decode(x_graph, x_syndrome_rounds)   # corrects Z errors
        z_correction = decode(z_graph, z_syndrome_rounds)   # corrects X errors
    elif decoder == "pymatching":
        x_correction = decode_pymatching(x_matching, x_graph, x_syndrome_rounds)
        z_correction = decode_pymatching(z_matching, z_graph, z_syndrome_rounds)
    else:
        raise ValueError(f"unknown decoder {decoder!r}")

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
    # tracked commutes with every stabilizer, so this is deterministic
    # under Pauli-only noise). expected hw_bit (before any correction)
    # == true_flip when starting from a deterministic +1 eigenstate of
    # the tracked logical operator AND no leakage (non-Pauli) event
    # occurred; leakage's `reset` is not a Pauli operator, so it breaks
    # the classical Pauli-frame's ability to predict hw_bit -- in that
    # case hw_bit (the real quantum state) is treated as ground truth
    # instead of asserting agreement with the (now invalid) frame.
    data_bits, _ = sample_and_project(sv, nq, DATA, rng)
    if logical_basis == "Z":
        hw_bit = sum(data_bits[d] for d in LOGICAL_Z) % 2
    else:
        hw_bit = sum(data_bits[d] for d in LOGICAL_X) % 2

    if any_leak:
        true_flip = hw_bit
    elif hw_bit != true_flip:
        raise AssertionError(
            f"HW/classical Pauli-frame mismatch: hw_bit={hw_bit} true_flip={true_flip} "
            "(statevector kernel and host bookkeeping disagree)")

    success = (true_flip == dec_flip)
    if stim_validator is not None and not stim_ok:
        raise AssertionError(
            f"Stim/FPGA ancilla-bit mismatch this trial "
            f"({stim_validator.mismatches} mismatches out of {stim_validator.checked} checked so far)")
    return TrialResult(x_error_qubits=[d for d in DATA if true_x_err[d]],
                        z_error_qubits=[d for d in DATA if true_z_err[d]],
                        x_correction=x_correction, z_correction=z_correction,
                        true_logical_flip=true_flip, hw_logical_bit=hw_bit,
                        decoded_logical_flip=dec_flip, success=success,
                        leaked=any_leak,
                        stim_mismatches=(stim_validator.mismatches if stim_validator else 0))


# =============================================================================
#  7. Self-test: pure-software verification of the lattice/decoder (no HW)
# =============================================================================
def _self_test_lattice(lattice: Lattice, label: str, check) -> None:
    """Algebraic + decoding-graph verification for one Lattice instance,
    generalized from the original 3x3-only checks so it can be run
    against DEFAULT_LATTICE, LARGER_LATTICE, or any build_lattice(dx,dy)
    instance. `check(name, cond)` is the shared pass/fail recorder from
    self_test()."""
    from qiskit.quantum_info import Pauli, Statevector, Operator

    nd = lattice.num_data

    def pauli_string(support: FrozenSet[int], letter: str) -> str:
        chars = ["I"] * nd
        for q in support:
            chars[q] = letter
        return "".join(reversed(chars))  # qiskit little-endian

    x_paulis = [Pauli(pauli_string(s, "X")) for s in lattice.x_supports]
    z_paulis = [Pauli(pauli_string(s, "Z")) for s in lattice.z_supports]
    xbar = Pauli(pauli_string(lattice.logical_x, "X"))
    zbar = Pauli(pauli_string(lattice.logical_z, "Z"))

    for i, a in enumerate(x_paulis):
        for j, b in enumerate(x_paulis):
            check(f"[{label}] XS{i} commutes XS{j}", a.commutes(b))
    for i, a in enumerate(z_paulis):
        for j, b in enumerate(z_paulis):
            check(f"[{label}] ZS{i} commutes ZS{j}", a.commutes(b))
    for i, a in enumerate(x_paulis):
        for j, b in enumerate(z_paulis):
            check(f"[{label}] XS{i} commutes ZS{j}", a.commutes(b))
    for i, a in enumerate(x_paulis):
        check(f"[{label}] Xbar commutes XS{i}", xbar.commutes(a))
        check(f"[{label}] Zbar commutes XS{i}", zbar.commutes(a))
    for i, a in enumerate(z_paulis):
        check(f"[{label}] Xbar commutes ZS{i}", xbar.commutes(a))
        check(f"[{label}] Zbar commutes ZS{i}", zbar.commutes(a))
    check(f"[{label}] Xbar anticommutes Zbar", not xbar.commutes(zbar))

    # code state: |0>^nd projected onto the +1 eigenspace of every
    # X-stabilizer via explicit operator projectors (pure linear algebra,
    # independent of the FPGA circuit), then verify every stabilizer and
    # Zbar read back out at the eigenvalues we expect.
    state = np.zeros(2 ** nd, dtype=complex)
    state[0] = 1.0
    ident = Operator(Pauli("I" * nd)).data
    for xp in x_paulis:
        proj = 0.5 * (ident + Operator(xp).data)
        state = proj @ state
        nrm = np.linalg.norm(state)
        if nrm > 1e-12:
            state = state / nrm
    sv = Statevector(state)
    for i, zp in enumerate(z_paulis):
        ev = sv.expectation_value(zp).real
        check(f"[{label}] code state: <ZS{i}> == +1", abs(ev - 1.0) < 1e-8)
    for i, xp in enumerate(x_paulis):
        ev = sv.expectation_value(xp).real
        check(f"[{label}] code state: <XS{i}> == +1", abs(ev - 1.0) < 1e-8)
    ev = sv.expectation_value(zbar).real
    check(f"[{label}] code state: <Zbar> == +1", abs(ev - 1.0) < 1e-8)

    # Decoder sanity, checked against the scientifically correct notion of
    # success ([12] Sec. III): a correction succeeds iff (true_error XOR
    # correction) has EVEN overlap with the logical operator's support --
    # i.e. it leaves the logical value unchanged -- not that it reproduces
    # the exact error location.
    graph = DecodeGraph(lattice.x_supports, rounds=1)
    for q in lattice.data:
        cur = [0] * len(lattice.x_supports)
        for i, s in enumerate(lattice.x_supports):
            if q in s:
                cur[i] ^= 1
        correction = decode(graph, [cur])
        residual = {q} ^ set(correction)
        logical_ok = (len(residual & lattice.logical_x) % 2) == 0
        check(f"[{label}] single Z-error on q{q}: decoder preserves logical Xbar "
              f"(correction={correction})", logical_ok)

    graph3 = DecodeGraph(lattice.z_supports, rounds=1)
    for q in lattice.data:
        cur = [0] * len(lattice.z_supports)
        for i, s in enumerate(lattice.z_supports):
            if q in s:
                cur[i] ^= 1
        correction = decode(graph3, [cur])
        residual = {q} ^ set(correction)
        logical_ok = (len(residual & lattice.logical_z) % 2) == 0
        check(f"[{label}] single X-error on q{q}: decoder preserves logical Zbar "
              f"(correction={correction})", logical_ok)


def self_test(verbose: bool = True) -> bool:
    ok = True

    def check(name, cond):
        nonlocal ok
        mark = "PASS" if cond else "FAIL"
        if verbose:
            print(f"  [{mark}] {name}")
        ok = ok and cond

    # --- default d=3 lattice: original hand-derived/hardcoded values ---
    _self_test_lattice(DEFAULT_LATTICE, "d3", check)

    # two far-apart single Z-errors decode independently (regression check
    # kept from the original self_test())
    graph2 = DecodeGraph(DEFAULT_LATTICE.x_supports, rounds=1)
    cur = [0] * len(DEFAULT_LATTICE.x_supports)
    for q in (0, 8):
        for i, s in enumerate(DEFAULT_LATTICE.x_supports):
            if q in s:
                cur[i] ^= 1
    correction = decode(graph2, [cur])
    check("[d3] two far-apart single Z-errors (q0,q8) decode to {q0,q8}",
          set(correction) == {0, 8})

    # --- build_lattice() consistency: general derivation reproduces the
    #     hand-derived 3x3 lattice exactly (as sets of stabilizer
    #     supports/logical operators, order-independent) ---
    rebuilt = build_lattice(3, 3)
    check("[build_lattice] 3x3 reproduces DEFAULT_LATTICE X-stabilizers",
          set(rebuilt.x_supports) == set(DEFAULT_LATTICE.x_supports))
    check("[build_lattice] 3x3 reproduces DEFAULT_LATTICE Z-stabilizers",
          set(rebuilt.z_supports) == set(DEFAULT_LATTICE.z_supports))
    check("[build_lattice] 3x3 reproduces DEFAULT_LATTICE Xbar/Zbar",
          rebuilt.logical_x == DEFAULT_LATTICE.logical_x and
          rebuilt.logical_z == DEFAULT_LATTICE.logical_z)

    # --- larger patch (future-work item (i)): same algebraic/decoding
    #     rigor applied to a genuinely different, bigger lattice ---
    check(f"[3x4] fits the {MAX_QUBITS_FPGA}-qubit budget "
          f"({LARGER_LATTICE.num_qubits} qubits)",
          LARGER_LATTICE.num_qubits <= MAX_QUBITS_FPGA)
    _self_test_lattice(LARGER_LATTICE, "3x4", check)

    # --- pymatching backend agreement with the brute-force solver
    #     (future-work item (iii)), skipped gracefully if not installed ---
    try:
        import pymatching  # noqa: F401
        pm_graph = DecodeGraph(DEFAULT_LATTICE.x_supports, rounds=1)
        pm_matching = build_pymatching_matching(pm_graph)
        agree = True
        for q in DEFAULT_LATTICE.data:
            cur = [0] * len(DEFAULT_LATTICE.x_supports)
            for i, s in enumerate(DEFAULT_LATTICE.x_supports):
                if q in s:
                    cur[i] ^= 1
            bf = set(decode(pm_graph, [cur]))
            pm = set(decode_pymatching(pm_matching, pm_graph, [cur]))
            same_logical = (len(({q} ^ pm) & DEFAULT_LATTICE.logical_x) % 2) == \
                           (len(({q} ^ bf) & DEFAULT_LATTICE.logical_x) % 2)
            agree = agree and same_logical
        check("[pymatching] agrees with brute-force MWPM on every single Z-error "
              "(same logical-Xbar outcome)", agree)
    except ImportError:
        if verbose:
            print("  [SKIP] pymatching not installed -- decoder cross-check skipped")

    # --- Stim cross-validation (future-work item (v)), skipped gracefully
    #     if not installed. Replay one synthetic round's gates on a fresh
    #     Stim TableauSimulator (data register starts |0>^n, exactly as
    #     round 0 of a real trial does) and postselect every Z-ancilla
    #     (deterministic from |0>^n, since Z|0>=|0>, so their outcome is
    #     fully pinned by the injected X errors alone -- unlike X-ancillas,
    #     which are genuinely random on round 0; see StimCrossValidator's
    #     docstring for why raw-bit comparison of X-ancillas would be a
    #     meaningless test here) to the classically-computed stabilizer
    #     parity: this must succeed (positive control). A deliberately
    #     wrong desired_value on the same qubit must then fail (negative
    #     control) -- proving postselect_z is actually constraining the
    #     state, not silently no-op'ing. ---
    try:
        import stim  # noqa: F401
        rng = np.random.default_rng(0)
        x_err_q, z_err_q, _ = sample_data_errors(rng, DEFAULT_LATTICE.data, 0.3, "independent", 0.0)
        gates = error_gates(x_err_q, z_err_q) + \
            syndrome_round_gates(DEFAULT_LATTICE, [], [])
        expected_z = {i: (len(s & set(x_err_q)) % 2) for i, s in enumerate(DEFAULT_LATTICE.z_supports)}

        validator = StimCrossValidator()
        validator.replay(gates)
        fpga_like_bits = {a: expected_z[i] for i, a in enumerate(DEFAULT_LATTICE.z_ancilla)}
        stim_ok = validator.check(DEFAULT_LATTICE.z_ancilla, fpga_like_bits)
        check("[stim] correct Z-ancilla parities postselect without error "
              "(positive control)", stim_ok and validator.mismatches == 0)

        bad_validator = StimCrossValidator()
        bad_validator.replay(gates)
        bad_bits = dict(fpga_like_bits)
        first_anc = DEFAULT_LATTICE.z_ancilla[0]
        bad_bits[first_anc] = 1 - bad_bits[first_anc]   # deliberately wrong
        bad_ok = bad_validator.check(DEFAULT_LATTICE.z_ancilla, bad_bits)
        check("[stim] a deliberately wrong Z-ancilla parity is caught as "
              "impossible (negative control)", not bad_ok and bad_validator.mismatches == 1)
    except ImportError:
        if verbose:
            print("  [SKIP] stim not installed -- Stim cross-validation self-test skipped")

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
    ap.add_argument("--dx", type=int, default=3, help="data-grid width (default 3x3 patch)")
    ap.add_argument("--dy", type=int, default=3, help="data-grid height (default 3x3 patch)")
    ap.add_argument("--decoder", choices=["bruteforce", "pymatching"], default="bruteforce",
                     help="matching-decoder backend (future-work item (iii))")
    ap.add_argument("--noise-model", choices=["independent", "depolarizing"], default="independent",
                     help="data-qubit error channel (future-work item (iv)); "
                          "'independent' reproduces [12]'s model unchanged")
    ap.add_argument("--leakage-prob", type=float, default=0.0,
                     help="per-qubit, per-round leakage-proxy probability (future-work item (iv), default off)")
    ap.add_argument("--stim-crossval", action="store_true",
                     help="cross-validate every round's ancilla bits against Stim (future-work item (v))")
    args = ap.parse_args()

    if args.selftest:
        ok = self_test()
        sys.exit(0 if ok else 1)

    lattice = DEFAULT_LATTICE if (args.dx, args.dy) == (3, 3) else build_lattice(args.dx, args.dy)

    print(f"[surface_code_qec] d={min(lattice.dx, lattice.dy)} rotated surface code "
          f"({lattice.dx}x{lattice.dy} patch): {lattice.num_data} data + "
          f"{lattice.num_ancilla} ancilla = {lattice.num_qubits} qubits (of {MAX_QUBITS_FPGA} available)")
    print(f"[surface_code_qec] decoder={args.decoder} noise-model={args.noise_model} "
          f"leakage-prob={args.leakage_prob} stim-crossval={args.stim_crossval}")
    print("[surface_code_qec] running self_test() before touching hardware...")
    if not self_test(verbose=False):
        print("[FATAL] self_test failed -- refusing to run on hardware.")
        sys.exit(1)
    print("[surface_code_qec] self_test PASS")

    if args.stim_crossval:
        try:
            import stim  # noqa: F401
        except ImportError:
            print("[FATAL] --stim-crossval requires the 'stim' package (pip install stim).")
            sys.exit(1)
    if args.decoder == "pymatching":
        try:
            import pymatching  # noqa: F401
        except ImportError:
            print("[FATAL] --decoder pymatching requires the 'pymatching' package (pip install pymatching).")
            sys.exit(1)

    fpga = QuantumFPGACircuit(args.xclbin)
    x_graph = DecodeGraph(lattice.x_supports, rounds=args.rounds)
    z_graph = DecodeGraph(lattice.z_supports, rounds=args.rounds)
    x_matching = build_pymatching_matching(x_graph) if args.decoder == "pymatching" else None
    z_matching = build_pymatching_matching(z_graph) if args.decoder == "pymatching" else None
    rng = np.random.default_rng(args.seed)

    rows = []
    total_leaked = 0
    total_stim_checked = 0
    total_stim_mismatches = 0
    for p in args.p:
        q_meas = args.q_meas if args.q_meas is not None else p
        n_fail = 0
        n_leaked = 0
        t0 = time.perf_counter()
        for t in range(args.trials):
            stim_validator = StimCrossValidator() if args.stim_crossval else None
            r = run_trial(fpga, lattice, args.rounds, p, q_meas, args.logical_basis, rng,
                          x_graph, z_graph, decoder=args.decoder,
                          x_matching=x_matching, z_matching=z_matching,
                          noise_model=args.noise_model, leak_p=args.leakage_prob,
                          stim_validator=stim_validator)
            if not r.success:
                n_fail += 1
            if r.leaked:
                n_leaked += 1
            if stim_validator is not None:
                total_stim_checked += stim_validator.checked
                total_stim_mismatches += stim_validator.mismatches
        dt = time.perf_counter() - t0
        rate = n_fail / args.trials
        total_leaked += n_leaked
        rows.append((p, args.trials, n_fail, rate))
        extra = f"  leaked={n_leaked}" if args.leakage_prob > 0 else ""
        print(f"  p={p:.4f}  trials={args.trials:4d}  logical_fail={n_fail:4d}  "
              f"logical_error_rate={rate:.4f}   ({dt:.1f}s, {dt/args.trials*1e3:.1f} ms/trial){extra}")

    if args.stim_crossval:
        print(f"[surface_code_qec] stim cross-validation: {total_stim_checked} ancilla bits checked, "
              f"{total_stim_mismatches} mismatches")
    if args.leakage_prob > 0:
        print(f"[surface_code_qec] leakage events: {total_leaked} of "
              f"{len(args.p) * args.trials} trials hit at least one leakage event")

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
                      f"({args.logical_basis}-bar, synthetic syndrome data, {lattice.num_qubits}-qubit FPGA statevector sim)")
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
