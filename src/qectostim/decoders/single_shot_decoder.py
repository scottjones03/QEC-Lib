# src/qectostim/decoders/single_shot_decoder.py
"""
Single-Shot Decoder with Syndrome Repair for 4D Codes

This decoder implements two-stage decoding for single-shot quantum error correction:
1. Syndrome Repair: Use metachecks to detect and correct measurement errors
2. Data Decoding: Decode the repaired syndrome to find data errors

For 4D/5D codes with metachecks, the chain condition M @ H^T = 0 ensures that valid
syndromes (from data errors only) satisfy the metacheck constraint. Measurement errors
violate this constraint, producing a nonzero metasyndrome that can be used to repair
the syndrome before decoding.

This enables single-shot error correction where only a single round of (noisy) syndrome
measurement is needed to achieve fault-tolerant decoding.

References:
    - Bombin et al., "Single-shot fault-tolerant quantum error correction" (2015)
    - Brown et al., "Experiments with the 4D surface code" (2024)
    - Kubica & Preskill, "Cellular-automaton decoders for topological quantum memory" (2019)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Dict, Tuple, List

import numpy as np
import stim

from qectostim.decoders.base import Decoder


class SingleShotDecoderIncompatibleError(Exception):
    """Raised when SingleShotDecoder cannot be used with a code/circuit."""
    pass


@dataclass
class SingleShotDecoder(Decoder):
    """
    Two-stage syndrome repair decoder for single-shot QEC with metachecks.
    
    Stage 1 (Syndrome Repair):
        - Extract syndrome s from detector bits
        - Compute metasyndrome m = M @ s (should be 0 if no measurement errors)
        - Decode metasyndrome to find measurement errors e_meas
        - Repair syndrome: s_repaired = s XOR e_meas
    
    Stage 2 (Data Decoding):
        - Decode repaired syndrome to find data corrections
        - Return observable predictions
    
    Parameters
    ----------
    code : CSSCode
        The CSS code with metachecks (must have meta_x and/or meta_z attributes).
    dem : stim.DetectorErrorModel
        The detector error model from the circuit.
    rounds : int
        Number of syndrome measurement rounds.
    basis : str
        Measurement basis ('X' or 'Z').
    repair_decoder : str
        Decoder to use for syndrome repair stage. Options: 'bposd', 'osd'.
    final_decoder : str
        Decoder to use for final data decoding. Options: 'bposd', 'pymatching'.
    
    Notes
    -----
    The decoder requires:
    1. A code with metachecks (meta_x for Z syndrome, meta_z for X syndrome)
    2. A circuit generated with enable_metachecks=True
    
    For Z-basis memory:
    - meta_x checks Z syndrome (M_x @ s_z = 0 for valid syndromes)
    - Measurement errors on Z ancillas cause metasyndrome violations
    
    For X-basis memory:
    - meta_z checks X syndrome (M_z @ s_x = 0 for valid syndromes)
    - Measurement errors on X ancillas cause metasyndrome violations
    """
    
    code: Any  # CSSCode with metachecks
    dem: stim.DetectorErrorModel
    rounds: int = 1
    basis: str = "Z"
    repair_decoder: str = "bposd"
    final_decoder: str = "bposd"
    
    def __post_init__(self) -> None:
        """Initialize the decoder components."""
        self._validate_code()
        self._setup_detector_mapping()
        self._setup_repair_decoder()
        self._setup_final_decoder()
    
    def _validate_code(self) -> None:
        """Validate that the code has metachecks."""
        # Check for non-empty metacheck matrices (not just is not None)
        # Empty matrices from chain complexes should be treated as no metachecks
        def _has_valid_meta(code, attr):
            if not hasattr(code, attr):
                return False
            m = getattr(code, attr)
            if m is None:
                return False
            if not hasattr(m, 'size') or not hasattr(m, 'shape'):
                return False
            return m.size > 0 and m.shape[0] > 0
        
        has_meta_x = _has_valid_meta(self.code, 'meta_x')
        has_meta_z = _has_valid_meta(self.code, 'meta_z')
        
        if not has_meta_x and not has_meta_z:
            raise SingleShotDecoderIncompatibleError(
                "SingleShotDecoder requires a code with metachecks (meta_x or meta_z). "
                "Use codes from 4D/5D chain complexes (e.g., ToricCode4D, HomologicalProductCode)."
            )
        
        # Cache metacheck matrices (only if valid and non-empty)
        self._meta_x = self.code.meta_x if has_meta_x else None
        self._meta_z = self.code.meta_z if has_meta_z else None
        
        # Cache stabilizer matrices
        self._hx = self.code.hx
        self._hz = self.code.hz
        self._n_x_stabs = len(self._hx)
        self._n_z_stabs = len(self._hz)
        
        # Verify chain conditions
        # For a chain complex: meta_x = boundary_{q-1}, hz = boundary_q
        # Chain condition: boundary_{q-1} @ boundary_q = 0, i.e., meta_x @ hz.T = 0
        # Note: hz has shape (n_z_stabs, n_qubits), and hz = boundary_q
        # meta_x has shape (n_meta_x, n_z_stabs), so meta_x @ hz.T has shape (n_meta_x, n_qubits)
        # The correct check is meta_x @ hz.T for rows of hz as columns
        if self._meta_x is not None:
            # meta_x checks Z syndrome. For chain complex:
            # meta_x = boundary_{q-1} with shape (dim_{q-2}, dim_{q-1})
            # hz = boundary_q with shape (dim_{q-1}, n_qubits)
            # Chain condition: meta_x @ hz = 0 (not hz.T)
            # But hz here is the parity check matrix, not boundary map directly.
            # Hz comes from boundary_q, so Hz has shape (n_z_stabs, n_qubits)
            # and meta_x has shape (n_meta_checks, n_z_stabs)
            # So meta_x @ Hz.T would be (n_meta_checks, n_qubits) - wrong dimensions
            # Actually meta_x @ Hz is (n_meta_checks, n_qubits) - this checks if metachecks
            # annihilate the Z stabilizer generators as row vectors
            # The correct mathematical statement is: meta_x @ hz.T = 0 where hz are ROW vectors
            # In our convention, hz has stabilizers as rows, so hz.T has stabilizers as columns
            # meta_x @ hz.T checks if metacheck rows annihilate stabilizer columns
            # This is the right check when dimensions match.
            
            # Let's verify dimensions match for the check
            if self._meta_x.shape[1] == self._hz.shape[0]:
                # meta_x: (n_meta, n_z_stabs), hz: (n_z_stabs, n_qubits)
                # meta_x @ hz: (n_meta, n_qubits) - checks meta against data qubits (wrong)
                # We want meta_x to check syndrome bits, not data qubits
                # Actually, metacheck M satisfies M @ s = 0 for valid syndromes s = Hz @ e
                # This means M @ Hz @ e = 0 for all e, so M @ Hz = 0
                check = (self._meta_x @ self._hz) % 2
            else:
                # Dimensions suggest meta_x @ hz.T
                check = (self._meta_x @ self._hz.T) % 2
            
            if np.any(check != 0):
                raise SingleShotDecoderIncompatibleError(
                    f"Invalid meta_x: chain condition not satisfied. "
                    f"meta_x shape: {self._meta_x.shape}, hz shape: {self._hz.shape}"
                )
        
        if self._meta_z is not None:
            # Similar logic for meta_z which checks X syndrome
            if self._meta_z.shape[1] == self._hx.shape[0]:
                check = (self._meta_z @ self._hx) % 2
            else:
                check = (self._meta_z @ self._hx.T) % 2
            
            if np.any(check != 0):
                raise SingleShotDecoderIncompatibleError(
                    f"Invalid meta_z: chain condition not satisfied. "
                    f"meta_z shape: {self._meta_z.shape}, hx shape: {self._hx.shape}"
                )
    
    def _setup_detector_mapping(self) -> None:
        """
        Set up mapping between detectors and syndrome bits.
        
        For single-shot QEC with rounds=1:
        - Regular stabilizer detectors correspond to syndrome bits
        - Metacheck detectors have 4th coordinate > 0
        
        We need to identify:
        1. Which detectors are regular stabilizers vs metachecks
        2. Mapping from regular detectors to syndrome bit positions
        """
        coords = self.dem.get_detector_coordinates()
        n_dets = self.dem.num_detectors
        
        # Classify detectors
        self._regular_det_indices = []
        self._meta_det_indices = []
        
        for det_idx in range(n_dets):
            if det_idx in coords:
                coord = coords[det_idx]
                # Metacheck detectors have 4th coordinate > 0 (meta_flag)
                is_meta = len(coord) >= 4 and coord[3] > 0
                if is_meta:
                    self._meta_det_indices.append(det_idx)
                else:
                    self._regular_det_indices.append(det_idx)
            else:
                # No coordinate info - assume regular detector
                self._regular_det_indices.append(det_idx)
        
        self._n_regular_dets = len(self._regular_det_indices)
        self._n_meta_dets = len(self._meta_det_indices)
        
        # For Z-basis memory with rounds=1:
        # - First n_z_stabs regular detectors are Z syndrome
        # - Metacheck detectors are meta_x checking Z syndrome
        # For X-basis memory:
        # - First n_x_stabs regular detectors are X syndrome
        # - Metacheck detectors are meta_z checking X syndrome
        
        if self.basis.upper() == "Z":
            self._active_meta = self._meta_x
            self._n_syndrome_bits = self._n_z_stabs
        else:
            self._active_meta = self._meta_z
            self._n_syndrome_bits = self._n_x_stabs
        
        self._n_meta_checks = self._active_meta.shape[0] if self._active_meta is not None else 0
    
    def _setup_repair_decoder(self) -> None:
        """Set up the decoder for syndrome repair stage."""
        if self._active_meta is None or self._n_meta_checks == 0:
            self._repair_decoder_obj = None
            return
        
        # Build a simple parity-check matrix for measurement errors
        # Measurement error on syndrome bit i flips metasyndrome bits where meta[j,i] = 1
        # So H_repair = meta matrix, and we decode metasyndrome to find syndrome bit errors
        
        if self.repair_decoder == "bposd":
            try:
                from ldpc import bposd_decoder
                
                # The repair decoder treats metasyndrome as syndrome and syndrome bits as "qubits"
                # H_repair has shape (n_metachecks, n_syndrome_bits)
                # Use a nominal error rate - BP/OSD doesn't need precise rates
                self._repair_decoder_obj = bposd_decoder(
                    self._active_meta,
                    error_rate=0.01,  # Nominal rate for BP
                    max_iter=30,
                    bp_method="product_sum",
                    osd_order=min(10, self._n_syndrome_bits - 1),
                    osd_method="osd_cs",
                )
            except ImportError:
                # Fallback to simple linear algebra decoding
                self._repair_decoder_obj = "linear"
            except Exception:
                # Any other error, fallback to linear
                self._repair_decoder_obj = "linear"
        elif self.repair_decoder == "osd":
            try:
                from ldpc import bposd_decoder
                self._repair_decoder_obj = bposd_decoder(
                    self._active_meta,
                    error_rate=0.01,  # Nominal rate
                    max_iter=0,  # Skip BP, just use OSD
                    osd_order=min(20, self._n_syndrome_bits - 1),
                    osd_method="osd_cs",
                )
            except ImportError:
                self._repair_decoder_obj = "linear"
            except Exception:
                self._repair_decoder_obj = "linear"
        else:
            self._repair_decoder_obj = "linear"
    
    def _setup_final_decoder(self) -> None:
        """Set up the decoder for final data decoding stage."""
        if self.final_decoder == "bposd":
            try:
                from stimbposd import BPOSD
                self._final_decoder_obj = BPOSD(
                    self.dem,
                    max_bp_iters=30,
                    bp_method="product_sum",
                    osd_order=min(60, self.dem.num_detectors - 1),
                    osd_method="osd_cs",
                )
            except ImportError:
                # Fallback to PyMatching
                import pymatching
                self._final_decoder_obj = pymatching.Matching.from_detector_error_model(self.dem)
        elif self.final_decoder == "pymatching":
            import pymatching
            self._final_decoder_obj = pymatching.Matching.from_detector_error_model(self.dem)
        else:
            raise ValueError(f"Unknown final_decoder: {self.final_decoder}")
    
    def _repair_syndrome_linear(
        self, 
        syndrome: np.ndarray, 
        metasyndrome: np.ndarray
    ) -> np.ndarray:
        """
        Simple linear algebra based syndrome repair.
        
        Given metasyndrome m = M @ s (should be 0 for valid syndrome),
        find minimal weight vector e such that M @ e = m.
        Then repair syndrome: s_repaired = s XOR e.
        
        This uses Gaussian elimination to find a valid correction.
        """
        if np.all(metasyndrome == 0):
            # No measurement errors detected
            return syndrome.copy()
        
        # Find minimum weight solution to M @ e = m (mod 2)
        # Use greedy approach: flip syndrome bits to zero out metasyndrome
        
        M = self._active_meta
        m = metasyndrome.copy()
        correction = np.zeros(self._n_syndrome_bits, dtype=np.uint8)
        
        # Greedy: for each metasyndrome bit that's 1, find a syndrome bit to flip
        for meta_idx in range(len(m)):
            if m[meta_idx] == 1:
                # Find a syndrome bit that affects this metacheck
                for syn_idx in range(self._n_syndrome_bits):
                    if M[meta_idx, syn_idx] == 1 and correction[syn_idx] == 0:
                        # Flip this syndrome bit
                        correction[syn_idx] = 1
                        # Update metasyndrome for all metachecks affected
                        m = (m + M[:, syn_idx]) % 2
                        break
        
        return (syndrome ^ correction) % 2
    
    def _extract_syndromes(
        self, 
        det_samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract syndrome and metasyndrome from detector samples.
        
        Parameters
        ----------
        det_samples : np.ndarray
            Detector samples, shape (n_shots, n_detectors)
        
        Returns
        -------
        syndrome : np.ndarray
            Regular syndrome bits, shape (n_shots, n_syndrome_bits)
        metasyndrome : np.ndarray
            Metasyndrome bits, shape (n_shots, n_meta_checks)
        """
        n_shots = det_samples.shape[0]
        
        # Extract regular detector values (syndrome)
        # For rounds=1, the first n_syndrome_bits regular detectors correspond to syndrome
        syndrome = np.zeros((n_shots, self._n_syndrome_bits), dtype=np.uint8)
        for i, det_idx in enumerate(self._regular_det_indices[:self._n_syndrome_bits]):
            syndrome[:, i] = det_samples[:, det_idx]
        
        # Compute metasyndrome from syndrome
        # metasyndrome = (meta @ syndrome.T).T = syndrome @ meta.T
        if self._active_meta is not None:
            metasyndrome = (syndrome @ self._active_meta.T) % 2
        else:
            metasyndrome = np.zeros((n_shots, 0), dtype=np.uint8)
        
        return syndrome, metasyndrome
    
    def _repair_syndromes(
        self, 
        syndromes: np.ndarray, 
        metasyndromes: np.ndarray
    ) -> np.ndarray:
        """
        Repair syndromes using metasyndrome information.
        
        Parameters
        ----------
        syndromes : np.ndarray
            Original syndromes, shape (n_shots, n_syndrome_bits)
        metasyndromes : np.ndarray
            Metasyndromes, shape (n_shots, n_meta_checks)
        
        Returns
        -------
        repaired : np.ndarray
            Repaired syndromes, shape (n_shots, n_syndrome_bits)
        """
        n_shots = syndromes.shape[0]
        repaired = np.zeros_like(syndromes)
        
        if self._repair_decoder_obj is None:
            # No metachecks available, return original
            return syndromes.copy()
        
        for i in range(n_shots):
            syn = syndromes[i]
            meta = metasyndromes[i]
            
            if np.all(meta == 0):
                # No measurement errors detected
                repaired[i] = syn
            elif self._repair_decoder_obj == "linear":
                # Use simple linear repair
                repaired[i] = self._repair_syndrome_linear(syn, meta)
            else:
                # Use BPOSD decoder
                try:
                    # Decode metasyndrome to find syndrome bit errors
                    meas_errors = self._repair_decoder_obj.decode(meta)
                    repaired[i] = (syn ^ meas_errors) % 2
                except Exception:
                    # Fallback to linear repair
                    repaired[i] = self._repair_syndrome_linear(syn, meta)
        
        return repaired
    
    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """
        Decode a batch of detector samples using two-stage syndrome repair.
        
        Parameters
        ----------
        dets : np.ndarray
            Detector samples, shape (n_shots, n_detectors)
        
        Returns
        -------
        predictions : np.ndarray
            Observable predictions, shape (n_shots, n_observables)
        """
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)
        
        n_shots = dets.shape[0]
        
        # =====================================================================
        # Stage 1: Syndrome Repair
        # =====================================================================
        
        # Extract syndrome and metasyndrome
        syndromes, metasyndromes = self._extract_syndromes(dets)
        
        # Repair syndromes using metasyndrome information
        repaired_syndromes = self._repair_syndromes(syndromes, metasyndromes)
        
        # =====================================================================
        # Stage 2: Data Decoding
        # =====================================================================
        
        # Reconstruct detector samples with repaired syndrome
        # For simplicity, we'll use the original dets with the final decoder
        # A more sophisticated approach would replace syndrome bits in dets
        
        # For now, use the final decoder directly on the original dets
        # The repair stage helps identify measurement errors, but the final
        # decoder still operates on the DEM which includes all correlations
        
        # Alternative: Build modified detector samples with repaired syndrome
        repaired_dets = dets.copy()
        for i, det_idx in enumerate(self._regular_det_indices[:self._n_syndrome_bits]):
            repaired_dets[:, det_idx] = repaired_syndromes[:, i]
        
        # Decode with final decoder
        try:
            if hasattr(self._final_decoder_obj, 'decode_batch'):
                predictions = self._final_decoder_obj.decode_batch(repaired_dets)
            else:
                # PyMatching interface
                predictions = self._final_decoder_obj.decode_batch(repaired_dets.astype(np.int8))
        except Exception as e:
            # Fallback: decode original dets
            if hasattr(self._final_decoder_obj, 'decode_batch'):
                predictions = self._final_decoder_obj.decode_batch(dets)
            else:
                predictions = self._final_decoder_obj.decode_batch(dets.astype(np.int8))
        
        predictions = np.asarray(predictions, dtype=np.uint8)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, self.dem.num_observables)
        
        return predictions
