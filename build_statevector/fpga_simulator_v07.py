
from __future__ import annotations
import math, os, time, uuid, logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import pyxrt as xrt
    XRT_AVAILABLE = True
except ImportError:
    XRT_AVAILABLE = False

try:
    from qiskit import QuantumCircuit, transpile
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    from qiskit.result import Result
    from qiskit.providers.models import QasmBackendConfiguration
    QISKIT_RESULT_AVAILABLE = True
except ImportError:
    QISKIT_RESULT_AVAILABLE = False

try:
    from qiskit_aer.version import __version__ as _aer_version
except ImportError:
    _aer_version = "0.14.0"

logger = logging.getLogger(__name__)

# =============================================================================
#  Constants
# =============================================================================
VERSION                = "07"
KERNEL_NAME            = "quantum_fpga_kernel_complex128"
# TODO: confirm against _x/reports/.../imp/impl_1_hw_bb_locked_timing_summary_routed.rpt
# for the actual build — 225 MHz was the [clock] freqHz target requested in
# alveo_u55c.cfg for this unified kernel; the link-stage log for this build
# reported per-domain clocks of KERNEL_CLK=500 MHz / DATA_CLK=300 MHz /
# hbm_aclk=450 MHz, which are not the same thing as the HLS kernel's own
# operating frequency. Update this constant once the routed timing report is
# checked so profiling estimates (EnergyEstimator, GateProfile) are accurate.
KERNEL_CLOCK_MHZ       = 225.0
KERNEL_CLOCK_PERIOD_NS = 1e3 / KERNEL_CLOCK_MHZ
# The compiled kernel (opt_kernel.cpp) is a 3-argument design: state_real,
# state_imag, gate_sequence -- each one flat, non-interleaved array, not an
# 8-way bank-interleaved layout. NUM_HBM_BANKS is kept only as descriptive
# metadata for how many HBM PCs the build config (build_statevector/
# alveo_u55c.cfg) spans across the two arrays + gate buffer (8+8+1=17), and
# is no longer used for any address-striping logic in this file.
NUM_HBM_BANKS          = 17
# 8 HBM PCs (256 MB each) per array per build_statevector/alveo_u55c.cfg ->
# 2 GB/array. 2^28 amps * 8 B = 2 GB exactly, so 28 qubits is the ceiling
# this config supports without widening the HBM range per array.
MAX_QUBITS_FPGA        = 28
GATE_WORDS             = 8
KERNEL_MAX_GATES       = 512
BYTES_PER_AMP          = 16       # complex128 = 2 × float64 (real + imag arrays combined)
BYTES_PER_F64          = 8

# =============================================================================
#  Kernel opcode table  (must match #define list in .cpp)
# =============================================================================
KERNEL_GATE_OPS: Dict[str, int] = {
    'id':0, 'h':1, 'x':2, 'y':3, 'z':4,
    's':5, 't':6, 'sdg':7, 'tdg':8, 'sx':9, 'sxdg':10,
    'rx':11, 'ry':12, 'rz':13,
    'p':14, 'phase':14, 'u1':14,
    'u2':15,
    'u3':16, 'u':16,
    'cx':17, 'cnot':17,
    'cz':18,
}

DECOMPOSABLE_GATES = frozenset({'mcx', 'mct'})
NON_UNITARY_GATES  = frozenset({'measure','barrier','snapshot','delay','initialize'})
RESET_GATE         = 'reset'

ALL_SUPPORTED_GATES: frozenset = frozenset({
    'id','h','x','y','z','s','t','sdg','tdg','sx','sxdg',
    'rx','ry','rz','p','phase','u1','u2','u3','u',
    'cx','cnot','cz',
    'cy','ch','swap','cp','cphase',
    'crx','cry','crz','csx','dcx',
    'iswap','ecr','rxx','ryy','rzz',
    'ccx','toffoli','ccnot',
    'mcx','mct',
})
TRUSTED_KERNEL_GATES_DEFAULT = ALL_SUPPORTED_GATES

# Power model constants (unchanged)
ALVEO_STATIC_W=28.0; ALVEO_HBM_IDLE_W=12.0
HBM_PJ_PER_BIT=3.9; LOGIC_MW_PER_MHZ_PER_KLUT=0.12; ACTIVE_KLUTS=82.4

# =============================================================================
#  Data classes
# =============================================================================
@dataclass
class GateProfile:
    gate_name:str; target_qubit:int; est_cycles:int; est_time_us:float
    hbm_read_kb:float; hbm_write_kb:float; est_energy_uj:float

@dataclass
class ValidationResult:
    passed:bool; norm:float; prob_sum:float; norm_ok:bool; prob_ok:bool
    expected_ok:Optional[bool]=None; expected_detail:str=""
    norm_tolerance:float=1e-6; prob_tolerance:float=1e-6

@dataclass
class RunProfile:
    circuit_name:str=""; num_qubits:int=0; num_gates:int=0
    num_chunks:int=1; num_resets:int=0
    host_distribute_us:float=0.0; host_encode_us:float=0.0
    xrt_kernel_us:float=0.0; host_collect_us:float=0.0
    host_reset_us:float=0.0; host_measure_us:float=0.0
    host_decompose_us:float=0.0; total_wall_us:float=0.0
    est_total_cycles:int=0; est_kernel_time_us:float=0.0
    total_hbm_read_mb:float=0.0; total_hbm_write_mb:float=0.0
    est_eff_bw_gbps:float=0.0; total_energy_mj:float=0.0
    avg_power_w:float=0.0; energy_per_gate_uj:float=0.0
    validation:Optional[ValidationResult]=None
    gate_profiles:List[GateProfile]=field(default_factory=list)
    gate_type_summary:Dict=field(default_factory=dict)

# =============================================================================
#  Validator
# =============================================================================
class FPGAValidator:
    NORM_TOLERANCE=1e-6; PROB_TOLERANCE=1e-6

    @staticmethod
    def check_norm(sv):
        n=float(np.linalg.norm(sv.astype(np.complex128,copy=False)))
        return abs(n-1.0)<=FPGAValidator.NORM_TOLERANCE, n

    @staticmethod
    def check_probability_sum(sv):
        s=sv.astype(np.complex128,copy=False)
        ps=float((s.real**2+s.imag**2).sum())
        return abs(ps-1.0)<=FPGAValidator.PROB_TOLERANCE, ps

    @staticmethod
    def check_deterministic(sv,idx,tolerance=0.02,label=""):
        p=float(sv[idx].real**2+sv[idx].imag**2); ok=p>=(1.0-tolerance)
        return ok,f"{label}:P(|{idx}>)={p:.8f} ({'PASS' if ok else 'FAIL'})"

    @staticmethod
    def check_uniform_superposition(sv,nq,tolerance=0.05,label=""):
        exp=1.0/math.sqrt(2**nq)
        dev=float(np.max(np.abs(np.abs(sv.astype(np.complex128,copy=False))-exp)))
        ok=dev<=tolerance
        return ok,f"{label}:max_dev={dev:.8f} ({'PASS' if ok else 'FAIL'})"

    @staticmethod
    def check_marginal_probability(sv,nq,pat,min_prob=0.5,label=""):
        N=2**nq; idx=np.arange(N,dtype=np.int64); mask=np.ones(N,dtype=bool)
        for q,b in pat.items(): mask&=(((idx>>q)&1)==b)
        s=sv.astype(np.complex128,copy=False)
        mp=float((s.real**2+s.imag**2)[mask].sum()); ok=mp>=min_prob
        return ok,f"{label}:P({pat})={mp:.8f} ({'PASS' if ok else 'FAIL'})"

    @classmethod
    def validate(cls,sv,nq,circuit_name="",det_idx=None,uniform_check=False,
                 marginal_pattern=None,marginal_min_prob=0.5,
                 det_tolerance=0.02,uniform_tolerance=0.05):
        no,n=cls.check_norm(sv); po,ps=cls.check_probability_sum(sv)
        eo,ed=None,""
        if det_idx is not None:
            eo,ed=cls.check_deterministic(sv,det_idx,tolerance=det_tolerance,label=circuit_name)
        elif marginal_pattern is not None:
            eo,ed=cls.check_marginal_probability(sv,nq,marginal_pattern,min_prob=marginal_min_prob,label=circuit_name)
        elif uniform_check:
            eo,ed=cls.check_uniform_superposition(sv,nq,tolerance=uniform_tolerance,label=circuit_name)
        passed=no and po and (eo is None or eo)
        return ValidationResult(passed=passed,norm=n,prob_sum=ps,norm_ok=no,prob_ok=po,
                                expected_ok=eo,expected_detail=ed,
                                norm_tolerance=cls.NORM_TOLERANCE,prob_tolerance=cls.PROB_TOLERANCE)

# =============================================================================
#  Energy estimator
# =============================================================================
class EnergyEstimator:
    @staticmethod
    def gate_energy_uj(cyc,hbm):
        dur=cyc/(KERNEL_CLOCK_MHZ*1e6)
        return(HBM_PJ_PER_BIT*hbm*8*1e-12+LOGIC_MW_PER_MHZ_PER_KLUT*KERNEL_CLOCK_MHZ*ACTIVE_KLUTS*1e-6*dur+ALVEO_STATIC_W*dur)*1e6

    @staticmethod
    def total_energy_mj(cyc,hbm):
        dur=cyc/(KERNEL_CLOCK_MHZ*1e6)
        if dur<=0: return 0.0,0.0
        tot=HBM_PJ_PER_BIT*hbm*8*1e-12+LOGIC_MW_PER_MHZ_PER_KLUT*KERNEL_CLOCK_MHZ*ACTIVE_KLUTS*1e-6*dur+(ALVEO_STATIC_W+ALVEO_HBM_IDLE_W)*dur
        return tot*1e3,tot/dur

# =============================================================================
#  MCX decomposition (Qiskit ancilla-free)
# =============================================================================
_MCX_CACHE: Dict[int,List[Tuple]]={}

def _decompose_mcx_qiskit(n_ctrl):
    if n_ctrl in _MCX_CACHE: return _MCX_CACHE[n_ctrl]
    if not QISKIT_AVAILABLE: raise RuntimeError(f"MCX n_ctrl={n_ctrl} requires Qiskit.")
    qc=QuantumCircuit(n_ctrl+1)
    try: qc.mcx(list(range(n_ctrl)),n_ctrl,mode='noancilla')
    except TypeError: qc.mcx(list(range(n_ctrl)),n_ctrl)
    basis=['ccx','cx','cz','h','t','tdg','s','sdg','x','y','z','rx','ry','rz','p','sx','sxdg','swap']
    comp=transpile(qc,basis_gates=basis,optimization_level=2)
    out=[]
    for inst in comp.data:
        nm=inst.operation.name.lower()
        if nm in NON_UNITARY_GATES or nm=='barrier': continue
        try: qs=[comp.find_bit(q).index for q in inst.qubits]
        except AttributeError: qs=[comp.qubits.index(q) for q in inst.qubits]
        ps=list(inst.operation.params) if inst.operation.params else None
        out.append((nm,qs,ps))
    _MCX_CACHE[n_ctrl]=out; return out

def _decompose_mcx(qubits):
    if not qubits: raise ValueError("mcx needs at least one qubit")
    tgt=int(qubits[-1]); ctrls=[int(q) for q in qubits[:-1]]; nc=len(ctrls)
    if tgt in ctrls: raise ValueError(f"mcx target {tgt} in controls {ctrls}")
    if nc==0: return[('x',[tgt],None)]
    if nc==1: return[('cx',[ctrls[0],tgt],None)]
    if nc==2: return[('ccx',[ctrls[0],ctrls[1],tgt],None)]
    lg=_decompose_mcx_qiskit(nc); qm=ctrls+[tgt]
    return[(nm,[qm[q] for q in lq],list(ps) if ps else None) for nm,lq,ps in lg]

# =============================================================================
#  Host-side gate lowering  →  {1Q native, CX, CZ} only
# =============================================================================
_PASSTHROUGH=frozenset({
    'id','h','x','y','z','s','t','sdg','tdg','sx','sxdg',
    'rx','ry','rz','p','phase','u1','u2','u3','u',
    'cx','cnot','cz',
    'measure','barrier','snapshot','delay','initialize','reset','mcx','mct',
})

def _lower_to_kernel_basis(gates):
    out=[]
    for g in gates:
        nm=g[0].lower() if isinstance(g[0],str) else g[0]
        q=g[1]; p=g[2] if len(g)>2 else None
        if nm in _PASSTHROUGH: out.append(g); continue
        if nm=='u': out.append(('u3',q,p)); continue
        if nm in('u1','phase'): out.append(('p',q,p)); continue
        if nm=='swap':
            a,b=int(q[0]),int(q[1])
            out.extend(_lower_to_kernel_basis([('cx',[a,b],None),('cx',[b,a],None),('cx',[a,b],None)])); continue
        if nm=='cy':
            c,t=int(q[0]),int(q[1])
            out.extend(_lower_to_kernel_basis([('sdg',[t],None),('cx',[c,t],None),('s',[t],None)])); continue
        if nm=='ch':
            c,t=int(q[0]),int(q[1])
            out.extend(_lower_to_kernel_basis([('ry',[t],[-math.pi/4]),('cz',[c,t],None),('ry',[t],[math.pi/4])])); continue
        if nm=='rzz':
            c,t=int(q[0]),int(q[1]); theta=float(p[0])
            out.extend(_lower_to_kernel_basis([('cx',[c,t],None),('rz',[t],[theta]),('cx',[c,t],None)])); continue
        if nm in('cp','cphase'):
            c,t=int(q[0]),int(q[1]); theta=float(p[0])
            out.append(('p',[c],[theta/2])); out.append(('p',[t],[theta/2]))
            out.extend(_lower_to_kernel_basis([('rzz',[c,t],[-theta/2])])); continue
        if nm=='crz':
            c,t=int(q[0]),int(q[1]); theta=float(p[0])
            out.extend(_lower_to_kernel_basis([('cx',[c,t],None),('rz',[t],[-theta/2]),('cx',[c,t],None),('rz',[t],[theta/2])])); continue
        if nm=='crx':
            c,t=int(q[0]),int(q[1]); theta=float(p[0])
            out.append(('h',[t],None))
            out.extend(_lower_to_kernel_basis([('crz',[c,t],[theta])]))
            out.append(('h',[t],None)); continue
        if nm=='cry':
            c,t=int(q[0]),int(q[1]); theta=float(p[0])
            out.extend(_lower_to_kernel_basis([('cx',[c,t],None),('ry',[t],[-theta/2]),('cx',[c,t],None),('ry',[t],[theta/2])])); continue
        if nm=='csx':
            c,t=int(q[0]),int(q[1])
            out.extend(_lower_to_kernel_basis([('crx',[c,t],[math.pi/2])]))
            out.append(('p',[c],[math.pi/4])); continue
        if nm=='dcx':
            a,b=int(q[0]),int(q[1])
            out.extend(_lower_to_kernel_basis([('cx',[a,b],None),('cx',[b,a],None)])); continue
        if nm=='iswap':
            a,b=int(q[0]),int(q[1])
            out.extend(_lower_to_kernel_basis([
                ('s',[a],None),('s',[b],None),('h',[a],None),
                ('cx',[a,b],None),('cx',[b,a],None),('h',[b],None)])); continue
        if nm=='ecr':
            a,b=int(q[0]),int(q[1])
            out.append(('h',[b],None))
            out.extend(_lower_to_kernel_basis([('cx',[a,b],None),('rz',[b],[math.pi/4]),('cx',[a,b],None)]))
            out.append(('h',[b],None)); out.append(('x',[a],None))
            out.append(('h',[b],None))
            out.extend(_lower_to_kernel_basis([('cx',[a,b],None),('rz',[b],[-math.pi/4]),('cx',[a,b],None)]))
            out.append(('h',[b],None)); continue
        if nm=='rxx':
            a,b=int(q[0]),int(q[1]); theta=float(p[0])
            out.append(('h',[a],None)); out.append(('h',[b],None))
            out.extend(_lower_to_kernel_basis([('cx',[a,b],None),('rz',[b],[theta]),('cx',[a,b],None)]))
            out.append(('h',[a],None)); out.append(('h',[b],None)); continue
        if nm=='ryy':
            a,b=int(q[0]),int(q[1]); theta=float(p[0])
            out.append(('sdg',[a],None)); out.append(('sdg',[b],None))
            out.append(('h',[a],None));   out.append(('h',[b],None))
            out.extend(_lower_to_kernel_basis([('cx',[a,b],None),('rz',[b],[theta]),('cx',[a,b],None)]))
            out.append(('h',[a],None)); out.append(('h',[b],None))
            out.append(('s',[a],None));   out.append(('s',[b],None)); continue
        if nm in('ccx','toffoli','ccnot'):
            a,b,c=int(q[0]),int(q[1]),int(q[2])
            out.extend(_lower_to_kernel_basis([
                ('h',[c],None),
                ('cx',[b,c],None),('tdg',[c],None),
                ('cx',[a,c],None),('t',[c],None),
                ('cx',[b,c],None),('tdg',[c],None),
                ('cx',[a,c],None),
                ('t',[b],None),('t',[c],None),
                ('h',[c],None),
                ('cx',[a,b],None),
                ('t',[a],None),('tdg',[b],None),
                ('cx',[a,b],None),
            ])); continue
        out.append(g)
    return out

# =============================================================================
#  Circuit parser / segment splitter
# =============================================================================
def _validate_and_split_gates(gates):
    """Split raw gate tuples into {segments, resets, meas}.

    meas is now a list of (qubit, clbit) pairs, in the order 'measure'
    instructions were encountered — NOT just a list of qubit indices. This
    is what lets one qubit fan out to several classical bits: a circuit that
    calls measure(0,0) and then measure(0,1) produces meas=[(0,0),(0,1)].

    A 'measure' gate tuple is (name, qubits, clbits). clbits may be None
    (or too short) for backward compatibility, in which case any qubit
    missing an explicit clbit maps to a clbit of the same index as the
    qubit — matching the old v06 default-measurement behaviour.
    """
    segments=[[]]; resets=[]; meas=[]
    for gt in gates:
        if len(gt)==2: nm,qs=gt; ps=None
        elif len(gt)==3: nm,qs,ps=gt
        else: raise ValueError(f"Bad gate tuple: {gt!r}")
        if not isinstance(nm,str): raise ValueError("Gate name must be str")
        name=nm.lower()
        if name in NON_UNITARY_GATES:
            if name=='measure':
                qlist=list(qs or [])
                clist=list(ps) if ps else []
                for i,q in enumerate(qlist):
                    qi=int(q)
                    ci=int(clist[i]) if i<len(clist) else qi
                    meas.append((qi,ci))
            continue
        if name==RESET_GATE:
            for q in(qs or []): resets.append((int(q),len(segments)-1))
            segments.append([]); continue
        if name not in ALL_SUPPORTED_GATES:
            raise ValueError(f"Unsupported gate '{nm}'")
        if name in DECOMPOSABLE_GATES:
            for sub in _decompose_mcx(list(qs)): segments[-1].append(sub)
        else:
            segments[-1].append((name,list(qs),ps))
    if segments and not segments[-1]: segments.pop()
    return segments,resets,meas

# =============================================================================
#  Core FPGA engine — 3-port (state_real/state_imag/gate_sequence) kernel version
# =============================================================================
class QuantumFPGACircuit:
    def __init__(self,xclbin_path,device_index=0):
        if not XRT_AVAILABLE: raise RuntimeError("pyxrt unavailable.")
        if not os.path.exists(xclbin_path): raise FileNotFoundError(f"XCLBIN not found: {xclbin_path}")
        self.xclbin_path=xclbin_path; self.device_index=device_index
        self._init_xrt()
        # state_real/state_imag BOs allocated lazily on first circuit run,
        # sized to actual nq. Pre-allocating 2^28 * 8 B per array at init
        # triggers "failed to allocate userptr bo: Operation not permitted"
        # — the kernel refuses to lock that much host memory before any
        # work is queued.
        #
        # The compiled kernel (opt_kernel.cpp) is a 3-argument design —
        # state_real, state_imag, gate_sequence — each its own flat,
        # non-interleaved array; there is no 8-way bank/index striping here
        # (that was the older, never-actually-deployed unified-kernel
        # protocol this driver used to speak, which does not match what
        # opt_kernel.cpp compiles to).
        self.state_real_bo=None; self.state_imag_bo=None
        self._current_array_bytes=0    # bytes per array currently allocated
        self._gate_bo=None; self._gate_bo_size=0
        self._sv_buffer=None; self._sv_buffer_n=0
        self._debug=bool(int(os.environ.get("FPGA_SIM_DEBUG","0") or "0"))
        logger.info(f"QuantumFPGACircuit v{VERSION} (complex128, 3-port real/imag) ready")

    def _init_xrt(self):
        try:
            self.device=xrt.device(self.device_index)
            self.xclbin_uuid=self.device.load_xclbin(self.xclbin_path)
            self.kernel=xrt.kernel(self.device,self.xclbin_uuid,KERNEL_NAME)
        except Exception as e: raise RuntimeError(f"FPGA init failed: {e}") from e

    def _ensure_hbm(self,nq):
        """Allocate (or grow) the state_real/state_imag BOs to fit nq qubits.
        Called before every distribute/collect operation so BO size always
        matches the circuit being executed. Avoids the large init-time
        allocation that causes 'failed to allocate userptr bo: Operation
        not permitted'."""
        N=2**nq
        bb=N*BYTES_PER_F64
        if bb<=self._current_array_bytes:
            return  # existing BOs are large enough
        # Kernel argument order is (state_real, state_imag, gate_sequence,
        # num_gates, num_qubits) -> group_id(0)/group_id(1) for the two
        # statevector arrays.
        self.state_real_bo=xrt.bo(self.device,bb,xrt.bo.flags.normal,self.kernel.group_id(0))
        self.state_imag_bo=xrt.bo(self.device,bb,xrt.bo.flags.normal,self.kernel.group_id(1))
        self._current_array_bytes=bb
        logger.debug(f"HBM re-allocated: {bb/1024:.1f} KB/array for nq={nq}")

    def _sv_buf(self,n):
        if self._sv_buffer is None or self._sv_buffer_n!=n:
            self._sv_buffer=np.zeros(n,dtype=np.complex128); self._sv_buffer_n=n
        else: self._sv_buffer.fill(0)
        return self._sv_buffer

    def _distribute_statevector(self,nq):
        self._ensure_hbm(nq)
        N=2**nq; db=N*BYTES_PER_F64
        vr=np.frombuffer(self.state_real_bo.map(),dtype=np.float64,count=N); vr.fill(0.0); vr[0]=1.0
        vi=np.frombuffer(self.state_imag_bo.map(),dtype=np.float64,count=N); vi.fill(0.0)
        self.state_real_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,db,0)
        self.state_imag_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,db,0)

    def _distribute_arbitrary_state(self,sv,nq):
        self._ensure_hbm(nq)
        N=2**nq; db=N*BYTES_PER_F64; sv128=sv.astype(np.complex128,copy=False)
        vr=np.frombuffer(self.state_real_bo.map(),dtype=np.float64,count=N); vr[:]=sv128.real
        vi=np.frombuffer(self.state_imag_bo.map(),dtype=np.float64,count=N); vi[:]=sv128.imag
        self.state_real_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,db,0)
        self.state_imag_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,db,0)

    def _collect_statevector(self,nq):
        self._ensure_hbm(nq)
        N=2**nq; db=N*BYTES_PER_F64
        sv=self._sv_buf(N)
        self.state_real_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,db,0)
        self.state_imag_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,db,0)
        sv.real[:]=np.frombuffer(self.state_real_bo.map(),dtype=np.float64,count=N)
        sv.imag[:]=np.frombuffer(self.state_imag_bo.map(),dtype=np.float64,count=N)
        return sv

    def _encode_gates(self,segment):
        n=len(segment)
        if n==0: return np.zeros(GATE_WORDS,dtype=np.int32),0,[]
        enc=np.zeros((n,GATE_WORDS),dtype=np.int32)
        for i,g in enumerate(segment):
            nm=g[0].lower() if isinstance(g[0],str) else g[0]
            qs=g[1]; ps=g[2] if len(g)>2 else None
            if nm not in KERNEL_GATE_OPS:
                raise ValueError(f"Gate '{nm}' not kernel-native — lowering bug.")
            enc[i,0]=KERNEL_GATE_OPS[nm]
            enc[i,1]=int(qs[0]) if len(qs)>0 else 0
            enc[i,2]=int(qs[1]) if len(qs)>1 else -1
            enc[i,3]=int(qs[2]) if len(qs)>2 else -1
            pf=np.zeros(3,dtype=np.float32)
            if ps:
                m=min(3,len(ps)); pf[:m]=[float(x) for x in ps[:m]]
            enc[i,4:7]=pf.view(np.int32)
        return enc.reshape(-1),n,list(segment)

    def _upload_gate_buffer(self,arr):
        nb=int(arr.nbytes)
        if self._gate_bo is None or self._gate_bo_size<nb:
            alloc=max(nb,4096)
            # gate_sequence is argument index 2 (state_real=0, state_imag=1,
            # gate_sequence=2) in the compiled 3-port kernel.
            self._gate_bo=xrt.bo(self.device,alloc,xrt.bo.flags.normal,self.kernel.group_id(2))
            self._gate_bo_size=alloc
        np.frombuffer(self._gate_bo.map(),dtype=np.int32,count=len(arr))[:]=arr
        self._gate_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,nb,0)

    def _launch_kernel(self,ng,nq):
        run=self.kernel(self.state_real_bo,self.state_imag_bo,self._gate_bo,ng,nq)
        run.wait()

    def _execute_segment(self,segment,nq):
        arr,nt,res=self._encode_gates(segment)
        xu=0.0; ch=0
        if nt==0: return xu,ch,res
        for s in range(0,nt,KERNEL_MAX_GATES):
            e=min(s+KERNEL_MAX_GATES,nt); nc=e-s
            chunk=arr[s*GATE_WORDS:e*GATE_WORDS]
            t0=time.perf_counter_ns()
            self._upload_gate_buffer(chunk); self._launch_kernel(nc,nq)
            xu+=(time.perf_counter_ns()-t0)/1e3; ch+=1
        return xu,ch,res

    def _apply_reset(self,qubit,nq):
        t0=time.perf_counter_ns()
        sv=self._collect_statevector(nq); N=2**nq
        mask=np.arange(N,dtype=np.int64)&(1<<qubit)
        sv.real[mask!=0]=0.0; sv.imag[mask!=0]=0.0
        nb=float(np.linalg.norm(sv))
        if nb<1e-12: sv.fill(0.0); sv[0]=1.0
        else: sv/=nb
        t1=time.perf_counter_ns()
        self._distribute_arbitrary_state(sv,nq)
        t2=time.perf_counter_ns()
        return(t1-t0)/1e3,(t2-t1)/1e3

    def _profile_gates(self,resolved,nq):
        N=2**nq; pairs=N>>1; out=[]
        for nm,qs,_ in resolved:
            q0=int(qs[0]) if qs else 0
            if nm in('cx','cnot','cz'): cyc=N*16; hbm=N*BYTES_PER_AMP
            else: cyc=pairs*16; hbm=pairs*2*BYTES_PER_AMP
            out.append(GateProfile(gate_name=nm,target_qubit=q0,est_cycles=cyc,
                est_time_us=cyc*KERNEL_CLOCK_PERIOD_NS/1e3,
                hbm_read_kb=hbm/2048,hbm_write_kb=hbm/2048,
                est_energy_uj=EnergyEstimator.gate_energy_uj(cyc,hbm)))
        return out

    def reset_state(self,nq): self._distribute_statevector(nq)

    def probe_native_gates(self,verbose=False):
        if verbose: print(f"  v{VERSION} native: {sorted(KERNEL_GATE_OPS)}")
        return frozenset(KERNEL_GATE_OPS.keys())

    def run_circuit(self,num_qubits,gates,shots=1024,validate=True,
                    det_idx=None,uniform_check=False,marginal_pattern=None,
                    marginal_min_prob=0.5,det_tolerance=0.02,uniform_tolerance=0.05,
                    circuit_name="",skip_init=False,num_clbits=None):
        """Run a circuit.

        num_clbits: width of the classical register the output bitstrings
        are built against. If None, defaults to num_qubits (matches v06
        behaviour for circuits with no explicit classical register). Any
        clbit never targeted by a 'measure' call stays '0' in every count
        key, and a qubit measured into several clbits (e.g. measure(0,0)
        then measure(0,1)) writes its sampled value into every one of them.
        """
        if not(1<=num_qubits<=MAX_QUBITS_FPGA):
            raise ValueError(f"num_qubits must be in [1,{MAX_QUBITS_FPGA}]")
        t0=time.perf_counter_ns(); N=2**num_qubits

        tlo=time.perf_counter_ns()
        segments,resets,meas=_validate_and_split_gates(gates)
        segments=[_lower_to_kernel_basis(seg) for seg in segments]
        hdu=(time.perf_counter_ns()-tlo)/1e3

        if not skip_init: self._distribute_statevector(num_qubits)
        td=time.perf_counter_ns()

        xu=0.0; ch=0; ng=0; ra=[]
        rm=defaultdict(list)
        for q,si in resets: rm[si].append(q)
        hru=0.0
        for si,seg in enumerate(segments):
            xu2,ch2,res=self._execute_segment(seg,num_qubits)
            xu+=xu2; ch+=ch2; ng+=len(res); ra.extend(res)
            for q in rm.get(si,[]):
                cu,ru=self._apply_reset(q,num_qubits); hru+=cu+ru
        tk=time.perf_counter_ns()

        sv=self._collect_statevector(num_qubits)
        tc=time.perf_counter_ns()
        nb=float(np.linalg.norm(sv))
        if nb>1e-12: sv/=nb

        val=None
        if validate:
            val=FPGAValidator.validate(sv=sv,nq=num_qubits,circuit_name=circuit_name,
                det_idx=det_idx,uniform_check=uniform_check,
                marginal_pattern=marginal_pattern,marginal_min_prob=marginal_min_prob,
                det_tolerance=det_tolerance,uniform_tolerance=uniform_tolerance)
            if not val.passed:
                logger.warning(f"[{circuit_name}] validation FAILED norm={val.norm:.10f}")

        gp=self._profile_gates(ra,num_qubits)
        tc2=sum(g.est_cycles for g in gp); th=sum((g.hbm_read_kb+g.hbm_write_kb)*1024 for g in gp)
        em,ap=EnergyEstimator.total_energy_mj(tc2,th)
        eb=(th/1e9)/(xu/1e6) if xu>0 else 0.0
        ts=defaultdict(lambda:{'count':0,'cycles':0,'energy_uj':0.0,'time_us':0.0})
        for g in gp:
            s=ts[g.gate_name]; s['count']+=1; s['cycles']+=g.est_cycles
            s['energy_uj']+=g.est_energy_uj; s['time_us']+=g.est_time_us

        # ── classical-bit-mapped measurement ──────────────────────────────
        nc = num_qubits if num_clbits is None else int(num_clbits)
        if nc<=0: nc = num_qubits
        meas_pairs = meas if meas else [(q,q) for q in range(min(num_qubits,nc))]

        probs=sv.real**2+sv.imag**2; ps=probs.sum()
        if ps>0: probs/=ps
        counts={}; memory=[]
        if shots>0:
            indices=np.random.choice(N,size=shots,p=probs)
            for idx in indices:
                qbits_str=format(int(idx),f'0{num_qubits}b')  # qubit q -> qbits_str[num_qubits-1-q]
                cbits=['0']*nc
                for q,c in meas_pairs:
                    if 0<=q<num_qubits and 0<=c<nc:
                        cbits[nc-1-c]=qbits_str[num_qubits-1-q]
                ob=''.join(cbits)
                counts[ob]=counts.get(ob,0)+1; memory.append(ob)
        te=time.perf_counter_ns()

        measured_qubits=sorted({q for q,_ in meas_pairs})

        prof=RunProfile(
            circuit_name=circuit_name,num_qubits=num_qubits,num_gates=ng,
            num_chunks=ch,num_resets=len(resets),
            host_distribute_us=(td-t0)/1e3,host_encode_us=0.0,
            xrt_kernel_us=xu,host_collect_us=(tc-tk)/1e3,
            host_reset_us=hru,host_measure_us=(te-tc)/1e3,
            host_decompose_us=hdu,total_wall_us=(te-t0)/1e3,
            est_total_cycles=tc2,est_kernel_time_us=tc2*KERNEL_CLOCK_PERIOD_NS/1e3,
            total_hbm_read_mb=sum(g.hbm_read_kb for g in gp)/1024,
            total_hbm_write_mb=sum(g.hbm_write_kb for g in gp)/1024,
            est_eff_bw_gbps=eb,total_energy_mj=em,avg_power_w=ap,
            energy_per_gate_uj=(em*1e3/ng if ng>0 else 0.0),
            validation=val,gate_profiles=gp,gate_type_summary=dict(ts))

        return{'statevector':sv.copy(),'counts':counts,'memory':memory,
               'measured_qubits':measured_qubits,'clbit_map':meas_pairs,'num_clbits':nc,
               'success':True,
               'execution_time':prof.total_wall_us/1e6,
               'timing':{'distribute_us':prof.host_distribute_us,
                         'decompose_us':prof.host_decompose_us,
                         'kernel_us':prof.xrt_kernel_us,
                         'collect_us':prof.host_collect_us,
                         'reset_us':prof.host_reset_us,
                         'measure_us':prof.host_measure_us,
                         'total_us':prof.total_wall_us,
                         'num_chunks':prof.num_chunks,
                         'num_resets':prof.num_resets},
               'profile':prof,'validation':val}

# =============================================================================
#  Qiskit-compatible wrapper
# =============================================================================
class FPGASimulator:
    """Qiskit-compatible FPGA backend — v07, complex128, 28-qubit, 8 HBM banks,
    explicit qubit→classical-bit measurement mapping."""
    _DEFAULT_XCLBIN="quantum_fpga_kernel_complex128.xclbin"

    def __init__(self,xclbin_path=None,device_index=0,**_):
        self._fpga=QuantumFPGACircuit(xclbin_path or self._DEFAULT_XCLBIN,device_index)
        self.name=f"fpga_simulator_v{VERSION}"; self._last_profile=None
        self._configuration=self._build_cfg() if QISKIT_RESULT_AVAILABLE else None

    @staticmethod
    def _build_cfg():
        return QasmBackendConfiguration(
            backend_name=f"fpga_simulator_v{VERSION}",backend_version=_aer_version,
            n_qubits=MAX_QUBITS_FPGA,
            basis_gates=sorted(ALL_SUPPORTED_GATES|{'measure','reset','barrier'}),
            gates=[],local=True,simulator=True,conditional=False,open_pulse=False,
            memory=True,max_shots=None,coupling_map=None,
            description=(f"FPGA v{VERSION} complex128 {MAX_QUBITS_FPGA}-qubit "
                         f"3-port (real/imag/gate_sequence) HBM kernel (SQ/CX/CZ only), "
                         f"explicit qubit-to-clbit measurement mapping"))

    def configuration(self): return self._configuration

    @property
    def last_profile(self): return self._last_profile

    def run(self,circuits,shots=1024,seed_simulator=None,**kwargs):
        t0=time.time()
        if not isinstance(circuits,list): circuits=[circuits]
        if seed_simulator is not None: np.random.seed(seed_simulator)
        results=[]
        for circuit in circuits:
            gates=[]; has_meas=False
            for inst in circuit.data:
                op=inst.operation.name
                try: qs=[circuit.find_bit(q).index for q in inst.qubits]
                except Exception: qs=[circuit.qubits.index(q) for q in inst.qubits]
                if op.lower()=='measure':
                    has_meas=True
                    try: cs=[circuit.find_bit(c).index for c in inst.clbits]
                    except Exception: cs=[circuit.clbits.index(c) for c in inst.clbits]
                    gates.append((op,qs,cs)); continue
                ps=getattr(inst.operation,'params',None)
                gates.append((op,qs,ps))
            if not has_meas and shots>0:
                qlist=list(range(circuit.num_qubits))
                gates.append(('measure',qlist,qlist))
            cname=getattr(circuit,'name','unnamed')
            nc=circuit.num_clbits if circuit.num_clbits>0 else circuit.num_qubits
            cr=self._fpga.run_circuit(circuit.num_qubits,gates,shots=shots,
                                       circuit_name=cname,num_clbits=nc)
            self._last_profile=cr['profile']; sv=cr.get('statevector'); pf=cr['profile']
            results.append({'success':True,'shots':shots,
                'data':{'counts':cr['counts'],
                        'statevector':sv.tolist() if isinstance(sv,np.ndarray) else None,
                        'execution_time':cr['execution_time']},
                'header':{'name':pf.circuit_name,
                    'metadata':{'fpga_enabled':True,'precision':'complex128',
                                'kernel_name':KERNEL_NAME,'hbm_banks':NUM_HBM_BANKS,
                                'kernel_clock_mhz':KERNEL_CLOCK_MHZ,
                                'execution_mode':f'xrt-hw-v{VERSION}-complex128-8bank',
                                'xrt_kernel_us':pf.xrt_kernel_us,
                                'num_chunks':pf.num_chunks,'num_resets':pf.num_resets,
                                'est_cycles':pf.est_total_cycles,
                                'energy_mj':pf.total_energy_mj,
                                'avg_power_w':pf.avg_power_w,
                                'eff_bw_gbps':pf.est_eff_bw_gbps,
                                'num_clbits':cr['num_clbits'],
                                'clbit_map':cr['clbit_map'],
                                'norm':pf.validation.norm if pf.validation else None,
                                'validation_passed':pf.validation.passed if pf.validation else None},
                    'timing':cr['timing'],
                    'memory_slots':cr['num_clbits'],
                    'qreg_sizes':[['q',circuit.num_qubits]],
                    'qubit_labels':[['q',i] for i in range(circuit.num_qubits)],
                    'creg_sizes':[['c',cr['num_clbits']]],
                    'clbit_labels':[['c',i] for i in range(cr['num_clbits'])]}})
        rd={'backend_name':self.name,'backend_version':_aer_version,'qobj_id':None,
            'job_id':kwargs.get('job_id',str(uuid.uuid4())),
            'success':all(r['success'] for r in results),'results':results,
            'date':datetime.now().isoformat(),'status':'COMPLETED','header':{},
            'metadata':{'shots':shots,'seed_simulator':seed_simulator,
                        'precision':'complex128','hbm_banks':NUM_HBM_BANKS,
                        'fpga_enabled':True,'total_run_time':time.time()-t0}}
        if QISKIT_RESULT_AVAILABLE:
            try: return Result.from_dict(rd)
            except Exception: pass
        class _R:
            def __init__(self,d): self._d=d
            def get_counts(self,*_): return self._d['results'][0]['data']['counts']
            def result(self): return self
        return _R(rd)

__version__=VERSION
__all__=['VERSION','KERNEL_NAME','KERNEL_GATE_OPS','ALL_SUPPORTED_GATES',
         'NUM_HBM_BANKS','MAX_QUBITS_FPGA','GATE_WORDS','KERNEL_MAX_GATES','BYTES_PER_AMP',
         'GateProfile','ValidationResult','RunProfile','FPGAValidator','EnergyEstimator',
         'QuantumFPGACircuit','FPGASimulator']
