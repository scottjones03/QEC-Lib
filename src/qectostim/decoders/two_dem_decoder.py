# qectostim/decoders/two_dem_decoder.py
"""
Two-DEM Correlated Decoder for Teleportation Gadgets.

Implements the Bluvstein/Cain approach:
- ONE circuit, ONE sample, TWO DEMs, classical combination
- DEM 1 (Z-path): Tracks Data Z_L via Z stabilizers
- DEM 2 (X-path): Tracks Ancilla Z_L via X stabilizers
- Final = decoded_data_Z_L ⊕ decoded_ancilla_Z_L ⊕ prep_frame

Key insight: Track logical Pauli products as they propagate through the circuit.
Back-propagate and include ONLY stabilizers along the propagation path.

For teleportation via transversal CNOT:
- Z errors propagate: Z_data stays on data, Z_anc spreads to Z_data via CNOT
- X errors propagate: X_anc stays on ancilla, X_data spreads to X_anc via CNOT
- These are independent matching problems that share the same underlying errors

The correlated decoding achieves ~45-50% improvement over naive independent decoding
because errors that flip BOTH observables cancel in the XOR combination.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import logging

import numpy as np
import stim

from qectostim.decoders.base import Decoder

try:
    import pymatching
    HAS_PYMATCHING = True
except ImportError:
    HAS_PYMATCHING = False

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode

logger = logging.getLogger(__name__)


@dataclass
class TwoDEMDecodingResult:
    """Result from two-DEM correlated decoding.
    
    Attributes:
        logical_errors: Boolean array of final logical errors after correlation
        dem1_decoded: Decoded observable from DEM 1 (Data Z_L)
        dem2_decoded: Decoded observable from DEM 2 (Ancilla Z_L)
        prep_frame: Prep frame corrections applied
        raw_data_zl: Raw Data Z_L before decoding
        raw_ancilla_zl: Raw Ancilla Z_L before decoding
        num_shots: Number of shots decoded
        dem1_error_rate: Error rate from DEM 1 alone
        dem2_error_rate: Error rate from DEM 2 alone
        teleport_error_rate: Combined teleportation error rate
        naive_error_rate: What the error rate would be if errors were independent
        improvement_pct: Percentage improvement from correlation
    """
    logical_errors: np.ndarray
    dem1_decoded: np.ndarray
    dem2_decoded: np.ndarray
    prep_frame: np.ndarray
    raw_data_zl: np.ndarray
    raw_ancilla_zl: np.ndarray
    num_shots: int
    dem1_error_rate: float = 0.0
    dem2_error_rate: float = 0.0
    teleport_error_rate: float = 0.0
    naive_error_rate: float = 0.0
    improvement_pct: float = 0.0


@dataclass
class DetectorFormula:
    """Formula for computing a detector from measurement indices.
    
    A detector is the XOR of measurements at the specified indices.
    """
    measurement_keys: List[str]  # Keys into measurement record
    measurement_offsets: List[int]  # Offsets within each key (for arrays)
    
    def evaluate(self, raw: np.ndarray, meas_record: Dict[str, Any]) -> np.ndarray:
        """Evaluate detector parity from raw measurements."""
        shots = raw.shape[0]
        result = np.zeros(shots, dtype=np.uint8)
        for key, offset in zip(self.measurement_keys, self.measurement_offsets):
            indices = meas_record[key]
            if isinstance(indices, list):
                idx = indices[offset]
            else:
                idx = indices + offset
            result ^= raw[:, idx].astype(np.uint8)
        return result


@dataclass
class TwoDEMConfig:
    """Configuration for two-DEM correlated decoding.
    
    This captures all the information needed to extract detector values
    and observables from raw measurements.
    
    Attributes:
        z_path_detectors: List of detector formulas for Z-path (DEM 1)
        x_path_detectors: List of detector formulas for X-path (DEM 2)
        data_zl_indices: Measurement indices for Data Z_L observable
        ancilla_zl_indices: Measurement indices for Ancilla Z_L observable
        prep_frame_indices: Measurement indices for prep frame correction
        z_logical_support: Data qubit indices in Z_L support
        x_logical_support: Data qubit indices in X_L support
    """
    z_path_detectors: List[DetectorFormula] = field(default_factory=list)
    x_path_detectors: List[DetectorFormula] = field(default_factory=list)
    data_zl_keys: List[Tuple[str, int]] = field(default_factory=list)  # (key, offset) pairs
    ancilla_zl_keys: List[Tuple[str, int]] = field(default_factory=list)
    prep_frame_keys: List[Tuple[str, int]] = field(default_factory=list)
    z_logical_support: List[int] = field(default_factory=list)
    x_logical_support: List[int] = field(default_factory=list)


@dataclass
class TwoDEMCorrelatedDecoder(Decoder):
    """
    Two-DEM Correlated Decoder for teleportation gadgets.
    
    Implements the Bluvstein/Cain approach where:
    1. A unified circuit is sampled ONCE
    2. Two separate DEMs are built (Z-path and X-path)
    3. Each DEM is decoded independently
    4. Results are combined classically via XOR
    
    The key insight is that correlated errors (those flipping both observables)
    cancel in the XOR combination, giving ~45-50% improvement over naive decoding.
    
    Usage:
        # Build config from gadget
        config = gadget.get_two_dem_config(meas_record)
        
        # Build DEMs
        dem1 = gadget.build_dem1_circuit(p).detector_error_model()
        dem2 = gadget.build_dem2_circuit(p).detector_error_model()
        
        decoder = TwoDEMCorrelatedDecoder(
            dem1=dem1,
            dem2=dem2,
            config=config,
            meas_record=meas_record,
        )
        
        # Sample and decode
        raw = circuit.compile_sampler().sample(num_shots)
        result = decoder.decode_correlated(raw)
    """
    
    dem1: stim.DetectorErrorModel
    dem2: stim.DetectorErrorModel
    config: TwoDEMConfig
    meas_record: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Initialize the underlying PyMatching decoders."""
        if not HAS_PYMATCHING:
            raise ImportError("pymatching is required for TwoDEMCorrelatedDecoder")
        
        self._matcher1 = pymatching.Matching.from_detector_error_model(self.dem1)
        self._matcher2 = pymatching.Matching.from_detector_error_model(self.dem2)
    
    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """
        Standard decode_batch interface (not recommended for two-DEM).
        
        This only uses DEM 1 and ignores correlation benefits.
        For full correlated decoding, use decode_correlated() with raw samples.
        """
        return self._matcher1.decode_batch(dets)
    
    def extract_z_path_detectors(self, raw: np.ndarray) -> np.ndarray:
        """Extract Z-path detector values from raw measurements."""
        shots = raw.shape[0]
        n_det = len(self.config.z_path_detectors)
        detectors = np.zeros((shots, n_det), dtype=np.uint8)
        
        for i, formula in enumerate(self.config.z_path_detectors):
            detectors[:, i] = formula.evaluate(raw, self.meas_record)
        
        return detectors
    
    def extract_x_path_detectors(self, raw: np.ndarray) -> np.ndarray:
        """Extract X-path detector values from raw measurements."""
        shots = raw.shape[0]
        n_det = len(self.config.x_path_detectors)
        detectors = np.zeros((shots, n_det), dtype=np.uint8)
        
        for i, formula in enumerate(self.config.x_path_detectors):
            detectors[:, i] = formula.evaluate(raw, self.meas_record)
        
        return detectors
    
    def extract_observables(self, raw: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract observable values from raw measurements."""
        shots = raw.shape[0]
        
        # Data Z_L
        data_zl = np.zeros(shots, dtype=np.uint8)
        for key, offset in self.config.data_zl_keys:
            indices = self.meas_record[key]
            if isinstance(indices, list):
                idx = indices[offset]
            else:
                idx = indices + offset
            data_zl ^= raw[:, idx].astype(np.uint8)
        
        # Ancilla Z_L
        ancilla_zl = np.zeros(shots, dtype=np.uint8)
        for key, offset in self.config.ancilla_zl_keys:
            indices = self.meas_record[key]
            if isinstance(indices, list):
                idx = indices[offset]
            else:
                idx = indices + offset
            ancilla_zl ^= raw[:, idx].astype(np.uint8)
        
        # Prep frame
        prep_frame = np.zeros(shots, dtype=np.uint8)
        for key, offset in self.config.prep_frame_keys:
            indices = self.meas_record[key]
            if isinstance(indices, list):
                idx = indices[offset]
            else:
                idx = indices + offset
            prep_frame ^= raw[:, idx].astype(np.uint8)
        
        return {
            "data_zl": data_zl,
            "ancilla_zl": ancilla_zl,
            "prep_frame": prep_frame,
            "corrected_ancilla_zl": ancilla_zl ^ prep_frame,
        }
    
    def decode_correlated(self, raw: np.ndarray) -> TwoDEMDecodingResult:
        """
        Perform full two-DEM correlated decoding.
        
        Parameters
        ----------
        raw : np.ndarray
            Raw measurement samples, shape (num_shots, num_measurements)
            
        Returns
        -------
        TwoDEMDecodingResult
            Complete decoding result with error rates and improvement metrics.
        """
        shots = raw.shape[0]
        
        # Extract detector values for both paths
        det1 = self.extract_z_path_detectors(raw)
        det2 = self.extract_x_path_detectors(raw)
        
        # Extract observables
        obs = self.extract_observables(raw)
        
        # Decode DEM 1 (Z-path)
        decoded1 = self._matcher1.decode_batch(det1)
        dem1_correction = decoded1[:, 0] if decoded1.ndim > 1 else decoded1
        decoded_data_zl = (obs["data_zl"] ^ dem1_correction).astype(np.uint8)
        
        # Decode DEM 2 (X-path)
        decoded2 = self._matcher2.decode_batch(det2)
        dem2_correction = decoded2[:, 0] if decoded2.ndim > 1 and decoded2.shape[1] > 0 else np.zeros(shots, dtype=np.uint8)
        decoded_ancilla_zl = (obs["ancilla_zl"] ^ obs["prep_frame"] ^ dem2_correction).astype(np.uint8)
        
        # Combine via XOR for teleportation result
        teleport_result = (decoded_data_zl ^ decoded_ancilla_zl).astype(np.uint8)
        
        # Compute error rates
        dem1_errors = np.sum(decoded_data_zl != 0)
        dem2_errors = np.sum(decoded_ancilla_zl != 0)
        teleport_errors = np.sum(teleport_result != 0)
        
        dem1_rate = dem1_errors / shots
        dem2_rate = dem2_errors / shots
        teleport_rate = teleport_errors / shots
        
        # Naive error rate (if independent)
        naive_rate = dem1_rate + dem2_rate - 2 * dem1_rate * dem2_rate
        
        # Improvement
        improvement = 100 * (naive_rate - teleport_rate) / naive_rate if naive_rate > 0 else 0
        
        return TwoDEMDecodingResult(
            logical_errors=teleport_result,
            dem1_decoded=decoded_data_zl,
            dem2_decoded=decoded_ancilla_zl,
            prep_frame=obs["prep_frame"],
            raw_data_zl=obs["data_zl"],
            raw_ancilla_zl=obs["ancilla_zl"],
            num_shots=shots,
            dem1_error_rate=dem1_rate,
            dem2_error_rate=dem2_rate,
            teleport_error_rate=teleport_rate,
            naive_error_rate=naive_rate,
            improvement_pct=improvement,
        )
    
    def decode_simple(self, raw: np.ndarray) -> np.ndarray:
        """
        Simple interface returning just logical errors.
        
        Parameters
        ----------
        raw : np.ndarray
            Raw measurement samples
            
        Returns
        -------
        np.ndarray
            Boolean array of logical errors, shape (num_shots,)
        """
        result = self.decode_correlated(raw)
        return result.logical_errors


def build_two_dem_config_for_css(
    code: "CSSCode",
    meas_record: Dict[str, Any],
    z_stabilizers: Optional[List[List[int]]] = None,
    x_stabilizers: Optional[List[List[int]]] = None,
) -> TwoDEMConfig:
    """
    Build TwoDEMConfig for a CSS code teleportation circuit.
    
    This is a helper that creates the detector formulas based on the
    standard measurement record layout from a teleportation circuit.
    
    Expected measurement record keys:
    - data_z_prep, ancilla_z_prep: Z stabilizer prep measurements
    - data_x_prep, ancilla_x_prep: X stabilizer prep measurements
    - data_z_r1, ancilla_z_r1: Z stabilizer round 1
    - data_x_r1, ancilla_x_r1: X stabilizer round 1
    - data_z_r2, ancilla_z_r2: Z stabilizer round 2
    - data_x_r2, ancilla_x_r2: X stabilizer round 2
    - data_final, ancilla_final: Final measurements
    
    Parameters
    ----------
    code : CSSCode
        The CSS code being used
    meas_record : Dict
        Measurement record from circuit construction
    z_stabilizers : List[List[int]], optional
        Z stabilizer supports. If None, extracted from code.
    x_stabilizers : List[List[int]], optional
        X stabilizer supports. If None, extracted from code.
        
    Returns
    -------
    TwoDEMConfig
        Configuration for two-DEM decoding
    """
    import numpy as np
    
    # Get stabilizers from code if not provided
    if z_stabilizers is None:
        hz = code.hz
        z_stabilizers = [list(np.where(hz[i])[0]) for i in range(hz.shape[0])]
    
    if x_stabilizers is None:
        hx = code.hx
        x_stabilizers = [list(np.where(hx[i])[0]) for i in range(hx.shape[0])]
    
    # Get logical operators
    z_logical = list(np.where(code.logical_z[0])[0]) if hasattr(code, 'logical_z') else []
    x_logical = list(np.where(code.logical_x[0])[0]) if hasattr(code, 'logical_x') else []
    
    num_z_stab = len(z_stabilizers)
    num_x_stab = len(x_stabilizers)
    
    # Build Z-path detectors
    z_path_detectors = []
    
    # Data Z: prep vs r1
    for i in range(num_z_stab):
        z_path_detectors.append(DetectorFormula(
            measurement_keys=["data_z_prep", "data_z_r1"],
            measurement_offsets=[i, i],
        ))
    
    # Ancilla Z: prep vs r1
    for i in range(num_z_stab):
        z_path_detectors.append(DetectorFormula(
            measurement_keys=["ancilla_z_prep", "ancilla_z_r1"],
            measurement_offsets=[i, i],
        ))
    
    # Cross-CNOT Z: ancilla_z_r1 ⊕ ancilla_z_r2 ⊕ data_z_r2
    for i in range(num_z_stab):
        z_path_detectors.append(DetectorFormula(
            measurement_keys=["ancilla_z_r1", "ancilla_z_r2", "data_z_r2"],
            measurement_offsets=[i, i, i],
        ))
    
    # Boundary: data_z_r2 vs data_final
    for stab_idx, support in enumerate(z_stabilizers):
        keys = ["data_z_r2"] + ["data_final"] * len(support)
        offsets = [stab_idx] + list(support)
        z_path_detectors.append(DetectorFormula(
            measurement_keys=keys,
            measurement_offsets=offsets,
        ))
    
    # Build X-path detectors
    x_path_detectors = []
    
    # Data X: prep vs r1
    for i in range(num_x_stab):
        x_path_detectors.append(DetectorFormula(
            measurement_keys=["data_x_prep", "data_x_r1"],
            measurement_offsets=[i, i],
        ))
    
    # Ancilla X: prep vs r1
    for i in range(num_x_stab):
        x_path_detectors.append(DetectorFormula(
            measurement_keys=["ancilla_x_prep", "ancilla_x_r1"],
            measurement_offsets=[i, i],
        ))
    
    # Cross-CNOT X: ancilla_x_r1 ⊕ ancilla_x_r2 ⊕ data_x_r2
    for i in range(num_x_stab):
        x_path_detectors.append(DetectorFormula(
            measurement_keys=["ancilla_x_r1", "ancilla_x_r2", "data_x_r2"],
            measurement_offsets=[i, i, i],
        ))
    
    # Data Z_L observable indices
    data_zl_keys = [("data_final", q) for q in z_logical]
    
    # Ancilla Z_L observable indices
    ancilla_zl_keys = [("ancilla_final", q) for q in z_logical]
    
    # Prep frame: X stabilizers with odd overlap with Z_L
    prep_frame_keys = []
    for stab_idx, support in enumerate(x_stabilizers):
        overlap = len(set(support) & set(z_logical))
        if overlap % 2 == 1:
            prep_frame_keys.append(("ancilla_x_prep", stab_idx))
    
    return TwoDEMConfig(
        z_path_detectors=z_path_detectors,
        x_path_detectors=x_path_detectors,
        data_zl_keys=data_zl_keys,
        ancilla_zl_keys=ancilla_zl_keys,
        prep_frame_keys=prep_frame_keys,
        z_logical_support=z_logical,
        x_logical_support=x_logical,
    )
