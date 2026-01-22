# src/qectostim/decoders/block_extraction.py
"""
Shared Block Extraction Utilities for Hierarchical/Concatenated Decoders.

This module provides common functionality for:
1. Computing detector slices for inner blocks and outer code
2. Collapsing time-like detector syndromes to stabilizer syndromes
3. Extracting code structure (inner/outer codes, logical supports)
4. Estimating physical error rates from DEMs

Used by:
- SoftMessagePassingDecoder
- TurboDecoderV2
- ExtrinsicTurboDecoder
- HierarchicalConcatenatedDecoder

Key Design Decisions:
--------------------
1. ADAPTIVE SLICING: Infers detector layout from DEM rather than assuming
   a specific FT circuit structure. This allows decoders to work with
   circuits from CSSMemoryExperiment, not just ConcatenatedMemoryExperiment.

2. ADAPTIVE SYNDROME COLLAPSE: Auto-detects whether detectors are hz-only,
   hx-only, or mixed, and collapses appropriately.

3. DUAL BASIS SUPPORT: Proper handling of both Z-basis and X-basis memories.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np

if TYPE_CHECKING:
    import stim
    from qectostim.codes.composite.concatenated import ConcatenatedCode


@dataclass
class CodeStructure:
    """
    Extracted structure from a concatenated code.
    
    Attributes
    ----------
    inner_code : Any
        The inner (physical) code.
    outer_code : Any
        The outer (logical) code.
    n_blocks : int
        Number of inner code blocks (= outer_code.n).
    inner_hx : np.ndarray
        X stabilizer parity check matrix for inner code.
    inner_hz : np.ndarray
        Z stabilizer parity check matrix for inner code.
    outer_hx : np.ndarray
        X stabilizer parity check matrix for outer code.
    outer_hz : np.ndarray
        Z stabilizer parity check matrix for outer code.
    inner_lx : np.ndarray
        Inner logical X operator support (binary vector).
    inner_lz : np.ndarray
        Inner logical Z operator support (binary vector).
    outer_lx : np.ndarray
        Outer logical X operator support (binary vector).
    outer_lz : np.ndarray
        Outer logical Z operator support (binary vector).
    """
    inner_code: Any
    outer_code: Any
    n_blocks: int
    inner_hx: np.ndarray
    inner_hz: np.ndarray
    outer_hx: np.ndarray
    outer_hz: np.ndarray
    inner_lx: np.ndarray
    inner_lz: np.ndarray
    outer_lx: np.ndarray
    outer_lz: np.ndarray


@dataclass
class DetectorSlices:
    """
    Detector index slices for hierarchical decoding.
    
    Attributes
    ----------
    inner_slices : Dict[int, Tuple[int, int]]
        Mapping from block_id to (start, stop) detector indices.
    outer_slice : Tuple[int, int]
        (start, stop) for outer code detectors.
    dets_per_block : int
        Number of detectors per inner block.
    total_dets : int
        Total number of detectors in DEM.
    """
    inner_slices: Dict[int, Tuple[int, int]]
    outer_slice: Tuple[int, int]
    dets_per_block: int
    total_dets: int


def extract_code_structure(code: 'ConcatenatedCode') -> CodeStructure:
    """
    Extract inner/outer codes and their parity check matrices.
    
    Handles various attribute naming conventions used by different code classes.
    
    Parameters
    ----------
    code : ConcatenatedCode
        The concatenated code.
        
    Returns
    -------
    CodeStructure
        Extracted code structure with all matrices.
    """
    # Get inner and outer codes (handle different attribute names)
    if hasattr(code, 'physical_code') and hasattr(code, 'logical_code'):
        inner_code = code.physical_code
        outer_code = code.logical_code
    elif hasattr(code, '_inner_code') and hasattr(code, '_outer_code'):
        inner_code = code._inner_code
        outer_code = code._outer_code
    elif hasattr(code, 'inner') and hasattr(code, 'outer'):
        inner_code = code.inner
        outer_code = code.outer
    else:
        raise ValueError(
            "Code must have inner/outer code attributes. "
            f"Got {type(code).__name__} with attributes: {dir(code)}"
        )
    
    n_blocks = outer_code.n
    
    # Get parity check matrices
    inner_hx = np.asarray(inner_code.hx, dtype=np.uint8)
    inner_hz = np.asarray(inner_code.hz, dtype=np.uint8)
    outer_hx = np.asarray(outer_code.hx, dtype=np.uint8)
    outer_hz = np.asarray(outer_code.hz, dtype=np.uint8)
    
    # Get logical supports
    inner_lx = get_logical_support(inner_code, 'X')
    inner_lz = get_logical_support(inner_code, 'Z')
    outer_lx = get_logical_support(outer_code, 'X')
    outer_lz = get_logical_support(outer_code, 'Z')
    
    return CodeStructure(
        inner_code=inner_code,
        outer_code=outer_code,
        n_blocks=n_blocks,
        inner_hx=inner_hx,
        inner_hz=inner_hz,
        outer_hx=outer_hx,
        outer_hz=outer_hz,
        inner_lx=inner_lx,
        inner_lz=inner_lz,
        outer_lx=outer_lx,
        outer_lz=outer_lz,
    )


def get_logical_support(code: Any, basis: str) -> np.ndarray:
    """
    Get binary support vector of the first logical operator.
    
    Handles multiple formats:
    - code.lz / code.lx (numpy array)
    - code.logical_z_ops / code.logical_x_ops (list of Pauli strings like 'ZZZIIII')
    - code.z_ops / code.x_ops (dict format {qubit: 'Z', ...})
    
    Parameters
    ----------
    code : Any
        Code object with logical operator attributes.
    basis : str
        'X' or 'Z' - which logical operator to get.
        
    Returns
    -------
    np.ndarray
        Binary vector indicating logical support.
    """
    n = code.n
    support = np.zeros(n, dtype=np.uint8)
    
    # Try array format first (code.lz / code.lx)
    if basis == 'Z':
        attr_names = ['lz', 'logical_z_ops', 'z_ops', 'logicals_z']
    else:
        attr_names = ['lx', 'logical_x_ops', 'x_ops', 'logicals_x']
    
    for attr in attr_names:
        if not hasattr(code, attr):
            continue
            
        val = getattr(code, attr)
        if val is None:
            continue
        
        # Handle callable (property or method)
        if callable(val):
            try:
                val = val()
            except Exception:
                continue
        
        # Handle different formats
        if isinstance(val, np.ndarray):
            arr = np.atleast_2d(val)
            if arr.shape[0] > 0 and arr.shape[-1] == n:
                support = (arr[0] != 0).astype(np.uint8)
                if np.sum(support) > 0:
                    return support
        
        elif isinstance(val, list) and len(val) > 0:
            first_op = val[0]
            
            # List of Pauli strings: ['ZZZIIII', ...]
            if isinstance(first_op, str):
                support = _pauli_string_to_support(first_op, n, basis)
                if np.sum(support) > 0:
                    return support
            
            # List of dicts: [{0: 'Z', 1: 'Z', ...}, ...]
            elif isinstance(first_op, dict):
                for qubit, pauli in first_op.items():
                    if isinstance(qubit, int) and qubit < n:
                        if basis == 'Z' and pauli in ('Z', 'z', 'Y', 'y'):
                            support[qubit] = 1
                        elif basis == 'X' and pauli in ('X', 'x', 'Y', 'y'):
                            support[qubit] = 1
                if np.sum(support) > 0:
                    return support
    
    # Fallback: assume transversal (all qubits in support)
    return np.ones(n, dtype=np.uint8)


def _pauli_string_to_support(pauli_str: str, n: int, target_pauli: str) -> np.ndarray:
    """
    Convert a Pauli string like 'ZZZIIII' to binary support vector.
    
    For target_pauli='Z', returns 1 where string has 'Z' or 'Y'.
    For target_pauli='X', returns 1 where string has 'X' or 'Y'.
    """
    support = np.zeros(n, dtype=np.uint8)
    
    for i, char in enumerate(pauli_str):
        if i >= n:
            break
        if target_pauli == 'Z' and char in ('Z', 'z', 'Y', 'y'):
            support[i] = 1
        elif target_pauli == 'X' and char in ('X', 'x', 'Y', 'y'):
            support[i] = 1
    
    return support


def compute_detector_slices(
    dem: 'stim.DetectorErrorModel',
    code_structure: CodeStructure,
    rounds: int,
    basis: str,
) -> DetectorSlices:
    """
    Compute detector slices for inner blocks and outer code.
    
    ADAPTIVE: Infers the actual detector layout from DEM rather than
    assuming a specific FT circuit structure. This allows decoders to
    work with circuits from CSSMemoryExperiment.
    
    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model.
    code_structure : CodeStructure
        Extracted code structure.
    rounds : int
        Number of syndrome measurement rounds.
    basis : str
        Measurement basis ('X' or 'Z').
        
    Returns
    -------
    DetectorSlices
        Detector slices for inner blocks and outer code.
    """
    total_dets = dem.num_detectors
    n_blocks = code_structure.n_blocks
    
    # Method 1: Assume equal distribution across blocks (most common)
    # This works when total_dets is evenly divisible by n_blocks
    dets_per_block_from_dem = total_dets // n_blocks if n_blocks > 0 else 0
    
    # Method 2: Calculate expected FT detector count using standard formula
    mx = code_structure.inner_hx.shape[0]  # X stabilizers
    mz = code_structure.inner_hz.shape[0]  # Z stabilizers
    
    if basis.upper() == "Z":
        x_dets_ft = max(0, rounds - 1) * mx
        z_dets_ft = (rounds + 1) * mz
    else:
        x_dets_ft = (rounds + 1) * mx
        z_dets_ft = max(0, rounds - 1) * mz
    dets_per_block_ft = x_dets_ft + z_dets_ft
    
    # Use actual DEM count if it makes sense, otherwise fall back to FT formula
    if (dets_per_block_from_dem * n_blocks == total_dets and 
        dets_per_block_from_dem > 0):
        dets_per_block = dets_per_block_from_dem
    elif dets_per_block_ft > 0:
        dets_per_block = dets_per_block_ft
    else:
        # Ultimate fallback: assume all detectors are inner, evenly distributed
        dets_per_block = total_dets // n_blocks if n_blocks > 0 else total_dets
    
    # CRITICAL: Cap dets_per_block to ensure inner blocks don't exceed total
    # This handles cases where formula-based estimate is larger than actual DEM
    max_inner_dets = total_dets  # Allow all detectors to be inner (no outer required)
    if n_blocks > 0 and dets_per_block * n_blocks > max_inner_dets:
        # Scale down to fit within available detectors
        dets_per_block = max_inner_dets // n_blocks
    
    # Build inner block slices (capped to not exceed total_dets)
    inner_slices = {}
    for i in range(n_blocks):
        start = min(i * dets_per_block, total_dets)
        stop = min((i + 1) * dets_per_block, total_dets)
        inner_slices[i] = (start, stop)
    
    # Outer detectors: whatever remains after inner blocks
    inner_total = min(n_blocks * dets_per_block, total_dets)
    outer_dets = max(0, total_dets - inner_total)
    outer_slice = (inner_total, inner_total + outer_dets)
    
    return DetectorSlices(
        inner_slices=inner_slices,
        outer_slice=outer_slice,
        dets_per_block=dets_per_block,
        total_dets=total_dets,
    )


def collapse_block_syndrome(
    det_syndrome: np.ndarray,
    n_hx: int,
    n_hz: int,
    rounds: int,
    basis: str,
) -> np.ndarray:
    """
    Collapse time-like detector syndrome to stabilizer syndrome.
    
    ADAPTIVE: Auto-detects the circuit structure from the number of detectors
    rather than assuming a specific FT protocol.
    
    Parameters
    ----------
    det_syndrome : np.ndarray
        Detector syndrome for a single block.
    n_hx : int
        Number of X stabilizers (rows of hx).
    n_hz : int
        Number of Z stabilizers (rows of hz).
    rounds : int
        Number of syndrome measurement rounds.
    basis : str
        Measurement basis ('X' or 'Z').
        
    Returns
    -------
    np.ndarray
        Stabilizer syndrome [hx_syn, hz_syn] where:
        - hx_syn detects Z errors
        - hz_syn detects X errors
    """
    n_det = len(det_syndrome)
    
    # Handle edge cases
    if n_det == 0:
        return np.concatenate([np.zeros(n_hx, dtype=np.uint8), 
                               np.zeros(n_hz, dtype=np.uint8)])
    
    # Case 1: Only hz detectors (Z-basis, hz-only measurement)
    if n_det > 0 and n_hz > 0 and n_det % n_hz == 0 and (n_hx == 0 or n_det % n_hx != 0 or n_hx == n_hz):
        n_rounds_hz = n_det // n_hz
        hz_stab_syn = np.zeros(n_hz, dtype=np.uint8)
        for t in range(n_rounds_hz):
            hz_stab_syn ^= det_syndrome[t * n_hz:(t + 1) * n_hz].astype(np.uint8)
        hx_stab_syn = np.zeros(n_hx, dtype=np.uint8)
        return np.concatenate([hx_stab_syn, hz_stab_syn])
    
    # Case 2: Only hx detectors (X-basis, hx-only measurement)
    if n_det > 0 and n_hx > 0 and n_det % n_hx == 0 and (n_hz == 0 or n_det % n_hz != 0):
        n_rounds_hx = n_det // n_hx
        hx_stab_syn = np.zeros(n_hx, dtype=np.uint8)
        for t in range(n_rounds_hx):
            hx_stab_syn ^= det_syndrome[t * n_hx:(t + 1) * n_hx].astype(np.uint8)
        hz_stab_syn = np.zeros(n_hz, dtype=np.uint8)
        return np.concatenate([hx_stab_syn, hz_stab_syn])
    
    # Case 3: Mixed hz and hx detectors - use the expected FT formula
    if basis.upper() == "Z":
        n_hz_dets = (rounds + 1) * n_hz
        n_hx_dets = max(0, rounds - 1) * n_hx
    else:
        n_hz_dets = max(0, rounds - 1) * n_hz
        n_hx_dets = (rounds + 1) * n_hx
    
    total_expected = n_hz_dets + n_hx_dets
    
    # Pad or truncate
    if n_det < total_expected:
        det_padded = np.zeros(total_expected, dtype=np.uint8)
        det_padded[:n_det] = det_syndrome
    else:
        det_padded = det_syndrome[:total_expected]
    
    # For Z-basis: hz detectors come first, then hx
    if basis.upper() == "Z":
        hz_det_start = 0
        hx_det_start = n_hz_dets
    else:
        hx_det_start = 0
        hz_det_start = n_hx_dets
    
    # Collapse hz detectors
    hz_stab_syn = np.zeros(n_hz, dtype=np.uint8)
    if n_hz_dets > 0 and n_hz > 0:
        n_hz_rounds = n_hz_dets // n_hz
        for t in range(n_hz_rounds):
            start = hz_det_start + t * n_hz
            end = hz_det_start + (t + 1) * n_hz
            hz_stab_syn ^= det_padded[start:end].astype(np.uint8)
    
    # Collapse hx detectors
    hx_stab_syn = np.zeros(n_hx, dtype=np.uint8)
    if n_hx_dets > 0 and n_hx > 0:
        n_hx_rounds = n_hx_dets // n_hx
        for t in range(n_hx_rounds):
            start = hx_det_start + t * n_hx
            end = hx_det_start + (t + 1) * n_hx
            hx_stab_syn ^= det_padded[start:end].astype(np.uint8)
    
    return np.concatenate([hx_stab_syn, hz_stab_syn])


def estimate_error_rate_from_dem(dem: 'stim.DetectorErrorModel', rounds: int = 1) -> float:
    """
    Estimate physical error rate from DEM.
    
    Parameters
    ----------
    dem : stim.DetectorErrorModel
        The detector error model.
    rounds : int
        Number of syndrome measurement rounds (for normalization).
        
    Returns
    -------
    float
        Estimated physical error rate.
    """
    total_prob = 0.0
    count = 0
    
    for instr in dem.flattened():
        if instr.type == "error":
            total_prob += instr.args_copy()[0]
            count += 1
    
    if count > 0:
        avg_prob = total_prob / count / max(1, rounds)
        return float(np.clip(avg_prob, 0.001, 0.1))
    return 0.01


def get_basis_matrices(
    code_structure: CodeStructure,
    basis: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Get the appropriate parity check and logical matrices for the given basis.
    
    For Z-basis memory:
    - X errors are detected by hz (Z stabilizers)
    - Z logical is flipped by X errors
    - Inner decoder uses hz, outer decoder uses hz
    
    For X-basis memory:
    - Z errors are detected by hx (X stabilizers)
    - X logical is flipped by Z errors
    - Inner decoder uses hx, outer decoder uses hx
    
    Parameters
    ----------
    code_structure : CodeStructure
        Extracted code structure.
    basis : str
        Measurement basis ('X' or 'Z').
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (H_inner, H_outer, L_inner, L_outer) where:
        - H_inner: parity check for inner decoder
        - H_outer: parity check for outer decoder
        - L_inner: inner logical support for soft XOR
        - L_outer: outer logical support
    """
    if basis.upper() == "Z":
        # Z-basis: X errors detected by hz, flip Z logical
        H_inner = code_structure.inner_hz
        H_outer = code_structure.outer_hz
        L_inner = code_structure.inner_lz
        L_outer = code_structure.outer_lz
    else:
        # X-basis: Z errors detected by hx, flip X logical
        H_inner = code_structure.inner_hx
        H_outer = code_structure.outer_hx
        L_inner = code_structure.inner_lx
        L_outer = code_structure.outer_lx
    
    return H_inner, H_outer, L_inner, L_outer


# =============================================================================
# LLR / Probability Conversions (shared across decoders)
# =============================================================================

def prob_to_llr(p: float) -> float:
    """
    Convert probability to log-likelihood ratio.
    
    LLR = log((1-p) / p)
    
    Positive LLR means p < 0.5 (likely no error).
    Negative LLR means p > 0.5 (likely error).
    """
    p = np.clip(p, 1e-10, 1 - 1e-10)
    return float(np.log((1 - p) / p))


def llr_to_prob(llr: float) -> float:
    """
    Convert log-likelihood ratio to probability.
    
    P(error=1) = 1 / (1 + exp(LLR))
    """
    llr = np.clip(llr, -30, 30)
    return float(1.0 / (1.0 + np.exp(llr)))


def llr_array_to_probs(llrs: np.ndarray) -> np.ndarray:
    """Convert array of LLRs to probabilities."""
    llrs_clipped = np.clip(llrs, -30, 30)
    return 1.0 / (1.0 + np.exp(llrs_clipped))


def probs_to_llr_array(probs: np.ndarray) -> np.ndarray:
    """Convert array of probabilities to LLRs."""
    probs_clipped = np.clip(probs, 1e-10, 1 - 1e-10)
    return np.log((1 - probs_clipped) / probs_clipped)


# =============================================================================
# Soft XOR Computation (shared across soft decoders)
# =============================================================================

def soft_xor(p_a: float, p_b: float) -> float:
    """
    Compute P(A XOR B = 1) from P(A=1) and P(B=1).
    
    The soft XOR formula:
        P(A ⊕ B = 1) = P(A)(1-P(B)) + P(B)(1-P(A))
    """
    return p_a * (1 - p_b) + p_b * (1 - p_a)


def soft_xor_over_support(probs: np.ndarray, support: np.ndarray) -> float:
    """
    Compute P(XOR = 1) over variables in support using soft XOR.
    
    P(L=1) = P(odd number of errors on L support)
    
    Uses the recurrence:
    P(x1 XOR x2 ... XOR xn = 1) computed iteratively
    
    Parameters
    ----------
    probs : np.ndarray
        Probability of error for each variable.
    support : np.ndarray
        Binary vector indicating which variables are in the XOR.
        
    Returns
    -------
    float
        Probability that XOR of supported variables equals 1.
    """
    p_xor_one = 0.0  # P(XOR so far = 1)
    
    for i, s in enumerate(support):
        if s and i < len(probs):
            p_i = probs[i]  # P(error at position i)
            # New P(XOR=1) = P(prev XOR=0)*P(this=1) + P(prev XOR=1)*P(this=0)
            p_xor_one = (1 - p_xor_one) * p_i + p_xor_one * (1 - p_i)
    
    return float(np.clip(p_xor_one, 1e-10, 1 - 1e-10))


def soft_xor_llr(qubit_llrs: np.ndarray, support: np.ndarray) -> float:
    """
    Compute LLR for logical operator L = XOR of supported qubits.
    
    Uses the soft XOR formula in LLR domain:
    LLR(XOR) = 2 * arctanh(prod(tanh(llr_i/2)))
    for all i in support.
    
    Parameters
    ----------
    qubit_llrs : np.ndarray
        LLR for each qubit.
    support : np.ndarray
        Binary vector indicating logical support.
        
    Returns
    -------
    float
        LLR for the logical operator.
    """
    prod = 1.0
    for i, s in enumerate(support):
        if s and i < len(qubit_llrs):
            prod *= np.tanh(qubit_llrs[i] / 2 + 1e-10)
    
    prod = np.clip(prod, -0.9999, 0.9999)
    return float(2 * np.arctanh(prod))


# =============================================================================
# Block Extraction Helper Class
# =============================================================================

class BlockExtractor:
    """
    Helper class for extracting block syndromes and computing soft information.
    
    This class encapsulates the common logic used by all hierarchical decoders:
    1. Slicing detector syndromes into per-block syndromes
    2. Collapsing time-like detectors to stabilizer syndromes
    3. Computing soft logical probabilities via BP + soft XOR
    
    Usage
    -----
    ```python
    extractor = BlockExtractor(code, dem, rounds=3, basis="Z")
    
    for block_id in range(extractor.n_blocks):
        block_syn = extractor.get_block_syndrome(full_syndrome, block_id)
        stab_syn = extractor.collapse_to_stabilizer_syndrome(block_syn)
        # ... run BP on stab_syn ...
    ```
    """
    
    def __init__(
        self,
        code: 'ConcatenatedCode',
        dem: 'stim.DetectorErrorModel',
        rounds: int = 3,
        basis: str = "Z",
    ):
        """
        Initialize the block extractor.
        
        Parameters
        ----------
        code : ConcatenatedCode
            The concatenated code.
        dem : stim.DetectorErrorModel
            Detector error model.
        rounds : int
            Number of syndrome measurement rounds.
        basis : str
            Measurement basis ('X' or 'Z').
        """
        self.code = code
        self.dem = dem
        self.rounds = rounds
        self.basis = basis.upper()
        self.num_detectors = dem.num_detectors
        
        # Extract code structure
        self.code_structure = extract_code_structure(code)
        self.n_blocks = self.code_structure.n_blocks
        
        # Get basis-specific matrices
        self.H_inner, self.H_outer, self.L_inner, self.L_outer = get_basis_matrices(
            self.code_structure, self.basis
        )
        
        # Also store full CSS matrices for soft decoders that need both
        self.inner_hx = self.code_structure.inner_hx
        self.inner_hz = self.code_structure.inner_hz
        self.inner_lx = self.code_structure.inner_lx
        self.inner_lz = self.code_structure.inner_lz
        
        # Compute detector slices
        self.slices = compute_detector_slices(dem, self.code_structure, rounds, basis)
        self.inner_slices = self.slices.inner_slices
        self.outer_slice = self.slices.outer_slice
        self.dets_per_block = self.slices.dets_per_block
        
        # Cache frequently used values
        self.n_inner_x_checks = self.inner_hx.shape[0]
        self.n_inner_z_checks = self.inner_hz.shape[0]
        self.n_inner_qubits = self.code_structure.inner_code.n
        self.n_outer_checks = self.H_outer.shape[0]
        
        # Estimate physical error rate
        self.p_channel = estimate_error_rate_from_dem(dem, rounds)
    
    def get_block_syndrome(
        self, 
        full_syndrome: np.ndarray, 
        block_id: int
    ) -> np.ndarray:
        """
        Extract detector syndrome for a specific inner block.
        
        Parameters
        ----------
        full_syndrome : np.ndarray
            Full detector syndrome.
        block_id : int
            Which inner block (0 to n_blocks-1).
            
        Returns
        -------
        np.ndarray
            Detector syndrome for the block.
        """
        det_start, det_stop = self.inner_slices[block_id]
        
        if det_stop <= len(full_syndrome):
            return full_syndrome[det_start:det_stop]
        else:
            # Pad with zeros if syndrome is shorter than expected
            block_syn = np.zeros(det_stop - det_start, dtype=np.uint8)
            if det_start < len(full_syndrome):
                available = len(full_syndrome) - det_start
                block_syn[:available] = full_syndrome[det_start:det_start + available]
            return block_syn
    
    def get_outer_syndrome(self, full_syndrome: np.ndarray) -> np.ndarray:
        """
        Extract detector syndrome for the outer code.
        
        Parameters
        ----------
        full_syndrome : np.ndarray
            Full detector syndrome.
            
        Returns
        -------
        np.ndarray
            Detector syndrome for outer code.
        """
        outer_start, outer_stop = self.outer_slice
        
        if outer_stop <= len(full_syndrome):
            return full_syndrome[outer_start:outer_stop]
        else:
            outer_syn = np.zeros(outer_stop - outer_start, dtype=np.uint8)
            if outer_start < len(full_syndrome):
                available = len(full_syndrome) - outer_start
                outer_syn[:available] = full_syndrome[outer_start:outer_start + available]
            return outer_syn
    
    def collapse_to_stabilizer_syndrome(
        self, 
        block_det_syndrome: np.ndarray
    ) -> np.ndarray:
        """
        Collapse block detector syndrome to stabilizer syndrome.
        
        Returns [hx_syn, hz_syn] format.
        
        Parameters
        ----------
        block_det_syndrome : np.ndarray
            Detector syndrome for a single block.
            
        Returns
        -------
        np.ndarray
            Stabilizer syndrome [hx_syn, hz_syn].
        """
        return collapse_block_syndrome(
            block_det_syndrome,
            self.n_inner_x_checks,
            self.n_inner_z_checks,
            self.rounds,
            self.basis,
        )
    
    def collapse_outer_syndrome(
        self,
        outer_det_syndrome: np.ndarray
    ) -> np.ndarray:
        """
        Collapse outer detector syndrome to stabilizer syndrome.
        
        Parameters
        ----------
        outer_det_syndrome : np.ndarray
            Detector syndrome for outer code.
            
        Returns
        -------
        np.ndarray
            Outer stabilizer syndrome.
        """
        n_stabs = self.n_outer_checks
        n_det = len(outer_det_syndrome)
        
        if n_det == 0:
            return np.zeros(n_stabs, dtype=np.uint8)
        
        if n_det == n_stabs:
            return outer_det_syndrome.astype(np.uint8)
        
        # XOR across rounds
        if n_stabs > 0 and n_det % n_stabs == 0:
            n_rounds_local = n_det // n_stabs
            result = np.zeros(n_stabs, dtype=np.uint8)
            for r in range(n_rounds_local):
                result ^= outer_det_syndrome[r * n_stabs:(r + 1) * n_stabs].astype(np.uint8)
            return result
        
        # Truncate or pad
        result = np.zeros(n_stabs, dtype=np.uint8)
        copy_len = min(n_det, n_stabs)
        result[:copy_len] = outer_det_syndrome[:copy_len]
        return result
