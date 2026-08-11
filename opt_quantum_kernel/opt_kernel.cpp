#include <ap_int.h>
#include <hls_math.h>

/* 
 * ===================================================================================
 * HARDWARE OPTIMIZED QUANTUM KERNEL (Vitis HLS)
 * ===================================================================================
 * 
 * MAJOR ARCHITECTURAL CHANGES FOR SYNTHESIS:
 * 
 * 1. Memory Interface (The Bank Conflict Fix):
 *    [SHORTCOMING OF OLD CODE]: The previous code used 8 separate double* arrays and 
 *    relied on a runtime `switch(bank)` statement to route data. In hardware, this 
 *    forces the compiler to build a massive dynamic multiplexer across all AXI memory 
 *    channels, preventing burst-reads and breaking pipelining (causing II=16).
 *    [NEW APPROACH]: We now use two contiguous arrays: `state_real` and `state_imag`. 
 *    We instruct the HLS compiler to use a 512-bit wide AXI bus (max_wstrb_bits=512). 
 *    This allows the FPGA to fetch eight 64-bit doubles in a single clock cycle natively, 
 *    without complex routing logic.
 * 
 * 2. Loop Pipelining (The II=1 Fix):
 *    [SHORTCOMING OF OLD CODE]: HLS compilers assume that reading and writing to the 
 *    same array in a loop creates a "carry dependency" (you might read a value you 
 *    just wrote), forcing the loop to stall (Initiation Interval II=16).
 *    [NEW APPROACH]: Added `#pragma HLS DEPENDENCE variable=... false`. Because 
 *    quantum state updates on distinct indices are mathematically independent per gate, 
 *    we can explicitly tell the compiler to ignore false dependencies, achieving II=1.
 * 
 * 3. Branch Elimination (The CX/CZ Fix):
 *    [SHORTCOMING OF OLD CODE]: The old CX/CZ loops iterated over the entire state space 
 *    (2^N) and used an `if` statement to skip 50% (or 75%) of the operations. FPGAs 
 *    do not have branch predictors like CPUs; an ignored loop cycle is a wasted cycle.
 *    [NEW APPROACH]: We now mathematically generate only the exact indices that need 
 *    modification using bitwise shifts. For example, CZ now loops exactly (2^N)/4 times 
 *    instead of 2^N times, executing 4x faster with zero branch logic.
 * ===================================================================================
 */

// ── Opcode constants ─────────────────────────────────────────
#define GATE_ID    0
#define GATE_H     1
#define GATE_X     2
#define GATE_Y     3
#define GATE_Z     4
#define GATE_S     5
#define GATE_T     6
#define GATE_SDG   7
#define GATE_TDG   8
#define GATE_SX    9
#define GATE_SXDG  10
#define GATE_RX    11
#define GATE_RY    12
#define GATE_RZ    13
#define GATE_P     14
#define GATE_U2    15
#define GATE_U3    16
#define GATE_CX    17
#define GATE_CZ    18

// ── Complex multiply (Inlined for DSP mapping) ───────────────
inline void cmul(double ar, double ai, double br, double bi,
                 double &rr, double &ri) {
    #pragma HLS INLINE
    rr = ar * br - ai * bi;
    ri = ar * bi + ai * br;
}

// ============================================================
//  FUNCTION 1 — apply_single_gate
// ============================================================
static void apply_single_gate(
    double* state_real, double* state_imag,
    int tgt, int nq,
    double g00r, double g00i,
    double g01r, double g01i,
    double g10r, double g10i,
    double g11r, double g11i
) {
    #pragma HLS INLINE off
    const int pairs  = 1 << (nq - 1);
    const int stride = 1 << tgt;

    SQ_LOOP: for (int p = 0; p < pairs; p++) {
        // Enforce 1 execution per clock cycle
        #pragma HLS PIPELINE II=1
        // Tell compiler that consecutive loop iterations do not depend on each other's memory writes
        #pragma HLS DEPENDENCE variable=state_real type=inter false
        #pragma HLS DEPENDENCE variable=state_imag type=inter false
        #pragma HLS LOOP_TRIPCOUNT min=1 max=268435456

        // Mathematically construct indices (No 'if' branches)
        int i0 = ((p >> tgt) << (tgt + 1)) | (p & (stride - 1));
        int i1 = i0 + stride;

        double a0r = state_real[i0];
        double a0i = state_imag[i0];
        double a1r = state_real[i1];
        double a1i = state_imag[i1];

        double t0r, t0i, t1r, t1i;
        
        cmul(g00r, g00i, a0r, a0i, t0r, t0i);
        cmul(g01r, g01i, a1r, a1i, t1r, t1i);
        double n0r = t0r + t1r;
        double n0i = t0i + t1i;

        cmul(g10r, g10i, a0r, a0i, t0r, t0i);
        cmul(g11r, g11i, a1r, a1i, t1r, t1i);
        double n1r = t0r + t1r;
        double n1i = t0i + t1i;

        state_real[i0] = n0r;
        state_imag[i0] = n0i;
        state_real[i1] = n1r;
        state_imag[i1] = n1i;
    }
}

// ============================================================
//  FUNCTION 2 — apply_cnot
// ============================================================
static void apply_cnot(double* state_real, double* state_imag, int ctrl, int tgt, int nq) {
    #pragma HLS INLINE off
    // A CNOT only affects pairs where the control bit is 1.
    // Instead of looping 2^N times and skipping half, we loop exactly (2^N)/2 times.
    const int N_half = 1 << (nq - 1);
    
    int min_bit = (ctrl < tgt) ? ctrl : tgt;
    int max_bit = (ctrl > tgt) ? ctrl : tgt;

    CX_LOOP: for (int p = 0; p < N_half; p++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=state_real type=inter false
        #pragma HLS DEPENDENCE variable=state_imag type=inter false
        #pragma HLS LOOP_TRIPCOUNT min=1 max=268435456

        // Reconstruct the index where control bit is forced to 1
        int p_low  = p & ((1 << min_bit) - 1);
        int p_mid  = (p >> min_bit) & ((1 << (max_bit - min_bit - 1)) - 1);
        int p_high = (p >> (max_bit - 1));
        
        int i = p_low | (p_mid << (min_bit + 1)) | (p_high << (max_bit + 1)) | (1 << ctrl);
        int j = i ^ (1 << tgt);

        // Hardware swap operation
        double ar = state_real[i];
        double ai = state_imag[i];
        double br = state_real[j];
        double bi = state_imag[j];

        state_real[i] = br;
        state_imag[i] = bi;
        state_real[j] = ar;
        state_imag[j] = ai;
    }
}

// ============================================================
//  FUNCTION 3 — apply_cz
// ============================================================
static void apply_cz(double* state_real, double* state_imag, int ctrl, int tgt, int nq) {
    #pragma HLS INLINE off
    // A CZ gate only flips the sign when BOTH control and target are 1.
    // We loop exactly (2^N)/4 times, eliminating 75% of dead clock cycles.
    const int N_quarter = 1 << (nq - 2);

    int min_bit = (ctrl < tgt) ? ctrl : tgt;
    int max_bit = (ctrl > tgt) ? ctrl : tgt;

    CZ_LOOP: for (int p = 0; p < N_quarter; p++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS DEPENDENCE variable=state_real type=inter false
        #pragma HLS DEPENDENCE variable=state_imag type=inter false
        #pragma HLS LOOP_TRIPCOUNT min=1 max=134217728

        int p_low  = p & ((1 << min_bit) - 1);
        int p_mid  = (p >> min_bit) & ((1 << (max_bit - min_bit - 1)) - 1);
        int p_high = (p >> (max_bit - 1));

        // Force both ctrl and tgt bits to 1
        int i = p_low | (p_mid << (min_bit + 1)) | (p_high << (max_bit + 1)) | (1 << ctrl) | (1 << tgt);

        double re = state_real[i];
        double im = state_imag[i];
        
        state_real[i] = -re;
        state_imag[i] = -im;
    }
}

// ============================================================
//  TOP-LEVEL KERNEL
// ============================================================
extern "C" void quantum_fpga_kernel_complex128(
    double* state_real, 
    double* state_imag,
    int*    gate_sequence,
    int     num_gates,
    int     num_qubits
) {
    // Utilize 512-bit wide memory ports to maximize PCIe/DDR bandwidth
    #pragma HLS INTERFACE m_axi port=state_real max_wstrb_bits=512 offset=slave bundle=gmem_real
    #pragma HLS INTERFACE m_axi port=state_imag max_wstrb_bits=512 offset=slave bundle=gmem_imag
    #pragma HLS INTERFACE m_axi port=gate_sequence offset=slave bundle=gmem_instr

    #pragma HLS INTERFACE s_axilite port=state_real    bundle=control
    #pragma HLS INTERFACE s_axilite port=state_imag    bundle=control
    #pragma HLS INTERFACE s_axilite port=gate_sequence bundle=control
    #pragma HLS INTERFACE s_axilite port=num_gates     bundle=control
    #pragma HLS INTERFACE s_axilite port=num_qubits    bundle=control
    #pragma HLS INTERFACE s_axilite port=return        bundle=control

    // ── Load gate sequence into BRAM ──────────────────────────
    int gbuf[512 * 8];
    #pragma HLS BIND_STORAGE variable=gbuf type=ram_2p impl=bram

    int total = num_gates * 8;
    LOAD: for (int i = 0; i < total; i++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=8 max=4096
        gbuf[i] = gate_sequence[i];
    }

    // ── Execute gates ─────────────────────────────────────────
    EXEC: for (int g = 0; g < num_gates; g++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=512

        int base = g * 8;
        int op   = gbuf[base];
        int q0   = gbuf[base + 1];
        int q1   = gbuf[base + 2];

        union { int i; float f; } p0, p1, p2;
        p0.i = gbuf[base + 4];
        p1.i = gbuf[base + 5];
        p2.i = gbuf[base + 6];
        
        double dp0 = (double)p0.f;
        double dp1 = (double)p1.f;
        double dp2 = (double)p2.f;

        if (op == GATE_ID) {
            continue;
        } else if (op == GATE_H) {
            const double h = 0.7071067811865476;
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                 h, 0.0,  h, 0.0,
                 h, 0.0, -h, 0.0);
        } else if (op == GATE_X) {
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                0.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 0.0, 0.0);
        } else if (op == GATE_Y) {
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                0.0,  0.0, 0.0, -1.0,
                0.0,  1.0, 0.0,  0.0);
        } else if (op == GATE_Z) {
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                1.0, 0.0,  0.0, 0.0,
                0.0, 0.0, -1.0, 0.0);
        } else if (op == GATE_S) {
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 1.0);
        } else if (op == GATE_T) {
            const double t = 0.7071067811865476;
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0,   t,   t);
        } else if (op == GATE_SDG) {
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                1.0, 0.0, 0.0,  0.0,
                0.0, 0.0, 0.0, -1.0);
        } else if (op == GATE_TDG) {
            const double t = 0.7071067811865476;
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0,   t,  -t);
        } else if (op == GATE_SX) {
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                0.5,  0.5,  0.5, -0.5,
                0.5, -0.5,  0.5,  0.5);
        } else if (op == GATE_SXDG) {
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                0.5, -0.5,  0.5,  0.5,
                0.5,  0.5,  0.5, -0.5);
        } else if (op == GATE_RX) {
            double c = hls::cos(dp0 * 0.5);
            double s = hls::sin(dp0 * 0.5);
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                  c, 0.0, 0.0,  -s,
                0.0,  -s,   c, 0.0);
        } else if (op == GATE_RY) {
            double c = hls::cos(dp0 * 0.5);
            double s = hls::sin(dp0 * 0.5);
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                c, 0.0, -s, 0.0,
                s, 0.0,  c, 0.0);
        } else if (op == GATE_RZ) {
            double c = hls::cos(dp0 * 0.5);
            double s = hls::sin(dp0 * 0.5);
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                c, -s, 0.0, 0.0,
                0.0, 0.0, c,  s);
        } else if (op == GATE_P) {
            double c = hls::cos(dp0);
            double s = hls::sin(dp0);
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0,   c,   s);
        } else if (op == GATE_U2) {
            const double inv_sqrt2 = 0.7071067811865476;
            double cphi = hls::cos(dp0), sphi = hls::sin(dp0);
            double clam = hls::cos(dp1), slam = hls::sin(dp1);
            double cpl  = hls::cos(dp0 + dp1);
            double spl  = hls::sin(dp0 + dp1);
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                 inv_sqrt2,              0.0,
                -inv_sqrt2 * clam, -inv_sqrt2 * slam,
                 inv_sqrt2 * cphi,  inv_sqrt2 * sphi,
                 inv_sqrt2 * cpl,   inv_sqrt2 * spl);
        } else if (op == GATE_U3) {
            double ct   = hls::cos(dp0 * 0.5), st = hls::sin(dp0 * 0.5);
            double cphi = hls::cos(dp1),       sphi = hls::sin(dp1);
            double clam = hls::cos(dp2),       slam = hls::sin(dp2);
            double cpl  = hls::cos(dp1 + dp2);
            double spl  = hls::sin(dp1 + dp2);
            apply_single_gate(state_real, state_imag, q0, num_qubits,
                  ct,          0.0,
                -st * clam,  -st * slam,
                 st * cphi,   st * sphi,
                 ct * cpl,    ct * spl);
        } else if (op == GATE_CX) {
            apply_cnot(state_real, state_imag, q0, q1, num_qubits);
        } else if (op == GATE_CZ) {
            apply_cz(state_real, state_imag, q0, q1, num_qubits);
        }
    }
}