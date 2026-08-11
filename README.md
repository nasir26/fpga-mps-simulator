# FPGA MPS Quantum Circuit Simulator

FPGA-accelerated quantum circuit simulators developed at C-DAC Noida, targeting
Xilinx Alveo U55C cards. Two simulator backends live in this repo:

- **Matrix Product State (MPS) simulator** (`fpga_mps_simulator.py`,
  `fpga_mps_simulator.cpp`, `mps_core.py`) — tensor-network simulation for
  large, low-entanglement circuits (up to ~4000 qubits at `chi_max<=64`),
  offloading 2-qubit gate contraction to an 8-HBM-bank FPGA kernel.
- **Statevector simulator** (`fpga_simulator_v07.py`, `opt_kernel.cpp`) —
  exact 28-qubit complex128 statevector simulation on a unified 8-bank
  FPGA kernel, used as ground truth / cross-check and for algorithms that
  need exact statevectors (Grover, QFT, surface-code QEC).

`mps_core.py` is a pure NumPy/SciPy reference implementation of the MPS
algorithms with no hardware dependency, used for correctness cross-checks
against the FPGA path.

Companion paper, research notes, and reference notebooks are kept in the
sibling repo: https://github.com/nasir26/fpga-mps-research

## Hardware / software prerequisites

- Xilinx Alveo U55C card(s), platform `xilinx_u55c_gen3x16_xdma_3_202210_1`
- Xilinx Vitis 2023.2 + XRT 2.x (for building/running the FPGA kernels)
- `pyxrt` (XRT's Python bindings) for host-side FPGA control
- Python 3.8 (developed/tested on 3.8.10)
- See `requirements.txt` for the full pinned Python dependency set
  (qiskit, qiskit-aer[-gpu], qiskit-nature, cuquantum-cu12, numpy, scipy,
  matplotlib, ...). GPU-related packages are only needed if you run the
  GPU reference comparisons (`gpu.py`, `gpu_aer.py`).

The pure-Python parts (`mps_core.py`, plotting/analysis scripts) run on any
platform without FPGA hardware; anything that imports `pyxrt` needs the
Xilinx runtime and an attached Alveo U55C.

## Setup

```bash
python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Building the FPGA kernels

Each kernel has its own build directory with a `run_build.sh` that invokes
Vitis `v++` compile + link against the Alveo U55C platform:

```bash
cd build_mps && ./run_build.sh              # -> fpga_mps_simulator.xclbin
cd build_statevector && ./run_build.sh       # -> quantum_fpga_kernel_complex128.xclbin
```

These require the Vitis 2023.2 toolchain and produce large binary
`.xclbin`/`.xo`/`.xsa` artifacts that are intentionally **not** tracked in
this repo (see `.gitignore`) — rebuild them from the `.cpp` kernel sources
(`fpga_mps_simulator.cpp`, `opt_kernel.cpp`) to reproduce a bitstream.

## Reproducing results

- `comprehensive_test_v07.py` — full gate-correctness and algorithm-scaling
  test suite for the statevector kernel (single/two-qubit native gates,
  decomposed multi-qubit gates, Grover/QFT scaling). Produces the
  `test_v07_*.png` plots and `graphs/test_v07_*.png`.
- `mcx_stress_v07.py` / `mcx.py` — multi-controlled-X stress tests, producing
  `mcx_stress_results.csv` and `mcx_stress_v07.png`.
- `surface_code_qec.py` — distance-3 rotated surface-code quantum-memory
  experiment on the statevector simulator, reproducing
  `surface_code_d3_results.csv` / `surface_code_d3_Z_results.csv` and the
  corresponding logical-error-rate plots. Method follows Dennis, Kitaev,
  Landahl, Preskill, "Topological quantum memory" (arXiv:quant-ph/0110143).
- `fpga_validate_nature_circuits.py` — cross-checks circuits from the Qiskit
  Nature tutorial notebooks (see the research repo's `qiskit_nature/`
  folder) against the real FPGA MPS simulator.
- `gpu_aer.py` / `gpu.py` / `aer_test.py` — Qiskit Aer / GPU reference
  comparisons used to validate FPGA results against established simulators.

All of the above assume an attached, built Alveo U55C kernel unless noted
otherwise; `mps_core.py`-only paths run without hardware.

## Repository layout

```
fpga_mps_simulator.{py,cpp}   MPS simulator: host driver + HLS kernel
mps_core.py                    Pure NumPy/SciPy MPS reference implementation
fpga_simulator_v07.py          Statevector simulator host driver
opt_kernel.cpp / opt_quantum_kernel/   Statevector HLS kernel sources
build_mps/, build_statevector/ Vitis v++ build configs + scripts (binaries gitignored)
graphs/                        Result plots
aux_files/                     Supporting data/utility files
*.py (test/validation scripts) Correctness, benchmark, and validation suites
```
