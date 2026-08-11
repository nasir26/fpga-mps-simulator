#!/bin/bash
set -e
set -o pipefail
cd "$(dirname "$0")"
source /tools/Xilinx/Vitis/2023.2/settings64.sh
if [ -f /opt/xilinx/xrt/setup.sh ]; then source /opt/xilinx/xrt/setup.sh; fi

PLATFORM=/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm
KNAME=fpga_mps_simulator
SRC=../fpga_mps_simulator.cpp

echo "[$(date)] === COMPILE (v++ -c) ==="
v++ -c -t hw \
    --platform "$PLATFORM" \
    -k "$KNAME" \
    --config alveo_u55c.cfg \
    --vivado.prop "run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}={-directive sdx_optimization_effort_high}" \
    -o "${KNAME}.xo" \
    "$SRC"

echo "[$(date)] === LINK (v++ -l) ==="
v++ -l -t hw \
    --platform "$PLATFORM" \
    --config alveo_u55c.cfg \
    --advanced.misc "report=type report_timing_summary name impl_report_timing_summary_route_design_summary steps {route_design} runs {impl_1} options {-max_paths 10}" \
    --advanced.misc "report=type report_timing_summary name impl_report_timing_summary_post_route_phys_opt_design_summary steps {post_route_phys_opt_design} runs {impl_1} options {-max_paths 10}" \
    -o "${KNAME}.xclbin" \
    "${KNAME}.xo"

echo "[$(date)] === BUILD COMPLETE ==="
ls -la "${KNAME}.xclbin"
