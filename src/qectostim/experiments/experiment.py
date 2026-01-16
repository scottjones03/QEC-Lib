"""Abstract experiment wrapper: compiles circuits and runs decoding."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import stim

from qectostim.codes.abstract_code import Code
from qectostim.decoders.decoder_selector import select_decoder
from qectostim.noise.models import NoiseModel

logger = logging.getLogger(__name__)

class Experiment(ABC):
    def __init__(
        self,
        code: Code,
        noise_model: Optional[NoiseModel] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.code = code
        self.noise_model = noise_model
        self.metadata = metadata or {}

    @abstractmethod
    def to_stim(self) -> stim.Circuit:
        ...

    def run_no_decode(
        self,
        shots: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Logical error estimator WITHOUT decoding (ideal measurement).

        Strategy:
          * Build circuit via self.to_stim()
          * Apply noise model (which inserts DEPOLARIZE gates, etc.)
          * Sample the full measurement record from circuit.compile_sampler()
          * Extract the OBSERVABLE_INCLUDE targets from the circuit
          * Use those targets to compute the logical observable from measurement results
          * Compare to ideal (0) to get logical error rate

        This uses the same observable definition as the DEM-based decoder,
        but doesn't attempt to correct detector errors. It's essentially
        "perfect syndrome extraction" without error correction.

        This is useful for understanding the "uncorrected" error rate when
        syndrome information is perfect but not used for correction.
        """
        # 1) Build circuit.
        circuit = self.to_stim()

        # 2) Apply noise model (adds DEPOLARIZE gates etc.).
        if self.noise_model is not None:
            circuit = self.noise_model.apply(circuit)

        # 3) Extract OBSERVABLE_INCLUDE targets from circuit.
        #    These tell us which measurement record indices to XOR together.
        observable_targets = []
        for inst in circuit:
            if inst.name == "OBSERVABLE_INCLUDE":
                # Targets are in the form of rec[-k] indices
                # We need to convert these to absolute indices
                for target in inst.targets_copy():
                    if target.is_measurement_record_target:
                        observable_targets.append(target.value)  # value is the negative lookback

        if not observable_targets:
            # Fallback: no observable in circuit, assume no errors
            logical_errors = np.zeros(shots, dtype=np.uint8)
            return {
                "shots": shots,
                "logical_errors": logical_errors,
                "logical_error_rate": float(logical_errors.mean()),
            }

        # 4) Sample the full measurement record.
        sampler = circuit.compile_sampler()
        meas = sampler.sample(shots)  # shape: (shots, num_measurements)
        num_meas = meas.shape[1]

        # 5) Convert relative indices to absolute indices.
        #    rec[-1] is the last measurement, rec[-2] is second-to-last, etc.
        abs_indices = []
        for rel_idx in observable_targets:
            # rel_idx is negative (e.g., -1, -2)
            abs_idx = num_meas + rel_idx
            if 0 <= abs_idx < num_meas:
                abs_indices.append(abs_idx)

        if not abs_indices:
            logical_errors = np.zeros(shots, dtype=np.uint8)
            return {
                "shots": shots,
                "logical_errors": logical_errors,
                "logical_error_rate": float(logical_errors.mean()),
            }

        # 6) Compute logical observable as XOR (parity) of selected measurement bits.
        #    (We assume ideal logical = 0 in both Z and X basis.)
        logical_vals = meas[:, abs_indices]  # shape: (shots, len(abs_indices))
        logical_observable = (logical_vals.sum(axis=1) % 2).astype(np.uint8)  # parity
        logical_errors = logical_observable  # 1 == logical flip (ideal was 0)

        return {
            "shots": shots,
            "logical_errors": logical_errors,
            "logical_error_rate": float(logical_errors.mean()),
        }

    def _get_code_distance(self) -> int:
        """Extract distance from code. Defaults to 3 if not available."""
        # Try metadata first
        if hasattr(self.code, 'metadata') and isinstance(self.code.metadata, dict):
            d = self.code.metadata.get('distance')
            if d is not None and isinstance(d, (int, float)):
                return int(d)
        # Try distance property
        if hasattr(self.code, 'distance'):
            d = self.code.distance
            if d is not None and isinstance(d, (int, float)):
                return int(d)
        # Try 'd' attribute directly (some codes use this)
        if hasattr(self.code, 'd'):
            d = self.code.d
            if d is not None and isinstance(d, (int, float)):
                return int(d)
        # Default to 3 (assume can correct)
        return 3


    def run_decode(
        self,
        shots: int = 10_000,
        decoder_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Distance-aware logical error estimator.

        Routes to either error detection (distance ≤2) or error correction (distance ≥3):
        
        For distance ≤2 (detection-only):
          - Circuit-based sampling with syndrome extraction
          - No decoder (impossible to correct at distance 2)
          - Returns: shots, logical_errors, logical_error_rate, 
                     syndrome_nonzero, undetected_errors, non_detection_rate, detection_efficiency
        
        For distance ≥3 (correction-capable):
          - DEM-based sampling with decoder
          - Full error correction pipeline
          - Returns: shots, logical_errors, logical_error_rate

        Common return keys: shots, logical_errors, logical_error_rate
        """
        distance = self._get_code_distance()
        logger.debug("Code distance: %d", distance)
        
        if distance <= 2:
            logger.debug("Distance %d <= 2: Using detection-only path", distance)
            return self._run_detection_path(shots)
        else:
            logger.debug("Distance %d >= 3: Using correction path", distance)
            return self._run_correction_path(shots, decoder_name)

    def _run_detection_path(self, shots: int = 10_000) -> Dict[str, Any]:
        """
        Error detection for low-distance codes (distance ≤ 2).
        
        Strategy:
          1. Build circuit and apply noise
          2. Sample detector and observable outcomes directly
          3. Calculate syndrome (any detector fired?)
          4. Calculate logical error (observable mismatch)
          5. Calculate detection efficiency: what fraction of logical errors were detected?
        """
        logger.debug("Starting detection path with shots=%d", shots)

        # 1) Build and apply noise
        base_circuit = self.to_stim()
        if self.noise_model is not None:
            circuit = self.noise_model.apply(base_circuit)
        else:
            circuit = base_circuit
        logger.debug("Circuit length: %d", len(circuit))

        # 2) Sample from circuit directly
        sampler = circuit.compile_sampler()
        samples = sampler.sample(shots=shots)

        # 3) Parse samples: handle both tuple and array formats
        if isinstance(samples, tuple):
            det_samples = np.asarray(samples[0], dtype=np.uint8)
            obs_samples = np.asarray(samples[1], dtype=np.uint8) if samples[1] is not None else None
        else:
            arr = np.asarray(samples, dtype=np.uint8)
            num_det = circuit.num_detectors
            num_obs = circuit.num_observables
            expected_cols = num_det + num_obs
            
            if arr.shape[1] >= expected_cols:
                det_samples = arr[:, :num_det]
                obs_samples = arr[:, num_det:expected_cols] if num_obs > 0 else None
            else:
                raise ValueError(f"Sampler array too small: expected at least {expected_cols}, got {arr.shape[1]}")

        logger.debug("det_samples.shape=%s, obs_samples.shape=%s", 
                     det_samples.shape, obs_samples.shape if obs_samples is not None else None)

        # 4) Calculate metrics
        # Syndrome is nonzero if ANY detector fired
        syndrome_nonzero = np.any(det_samples != 0, axis=1).astype(np.uint8)
        
        # Observable error: compare to ideal (which would be all 0s for |0> state)
        logical_errors = obs_samples[:, 0] if obs_samples is not None else np.zeros(shots, dtype=np.uint8)
        
        # Undetected errors: logical error occurred but syndrome was zero
        undetected = (logical_errors == 1) & (syndrome_nonzero == 0)
        undetected_count = np.sum(undetected)
        
        # Total logical errors
        logical_error_count = np.sum(logical_errors)
        
        # Detection efficiency: of errors that occurred, how many were detected?
        if logical_error_count > 0:
            detection_efficiency = (logical_error_count - undetected_count) / logical_error_count
        else:
            detection_efficiency = 1.0  # No errors to detect
        
        logger.debug("Detection results: logical_errors=%d, syndrome_nonzero=%d, undetected=%d, efficiency=%.4f, ler=%.6f",
                     int(logical_error_count), int(np.sum(syndrome_nonzero)), int(undetected_count),
                     float(detection_efficiency), float(logical_error_count / shots))

        return {
            'shots': shots,
            'logical_errors': logical_errors,
            'logical_error_rate': float(logical_error_count / shots),
            'syndrome_nonzero': int(np.sum(syndrome_nonzero)),
            'undetected_errors': int(undetected_count),
            'non_detection_rate': float(undetected_count / shots),
            'detection_efficiency': float(detection_efficiency),
        }

    def _run_correction_path(
        self,
        shots: int = 10_000,
        decoder_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Error correction for high-distance codes (distance ≥ 3).
        
        Strategy:
          1. Build the (possibly noisy) circuit.
          2. Extract a DetectorErrorModel (DEM).
          3. Build a decoder from the DEM (e.g. PyMatchingDecoder).
          4. Sample *directly* from the DEM:
               - detector outcomes
               - logical observable bits
          5. Run the decoder on detector outcomes to get predicted logical flips.
          6. Compare predicted vs. true logical bits to estimate logical error rate.
        """
        logger.debug("Starting correction path with shots=%d, decoder=%s", shots, decoder_name)

        # 1) Build base circuit from the experiment.
        base_circuit = self.to_stim()
        logger.debug("Base circuit: %d instructions", len(base_circuit))

        # 2) Apply noise model (if any).
        if self.noise_model is not None:
            circuit = self.noise_model.apply(base_circuit)
        else:
            circuit = base_circuit
        logger.debug("Noisy circuit: %d instructions", len(circuit))

        # 3) Build DetectorErrorModel from noisy circuit.
        # For color codes and other hypergraph codes, we need to handle decomposition failures
        # by using ignore_decomposition_failures=True
        try:
            dem = circuit.detector_error_model(decompose_errors=True)
        except ValueError as e:
            if "Failed to decompose errors" in str(e):
                # This is a hypergraph code (e.g., color code) - use ignore_decomposition_failures
                logger.debug("Hypergraph DEM detected, using ignore_decomposition_failures")
                dem = circuit.detector_error_model(
                    decompose_errors=True,
                    ignore_decomposition_failures=True
                )
            else:
                raise
        logger.debug("DEM: %d detectors, %d errors, %d observables", 
                     dem.num_detectors, dem.num_errors, dem.num_observables)

        if dem.num_observables == 0:
            logger.warning("DEM has no observables; returning zero logical errors")
            logical_errors = np.zeros(shots, dtype=np.uint8)
            return {
                "shots": shots,
                "logical_errors": logical_errors,
                "logical_error_rate": float(logical_errors.mean()),
            }

        # 4) Build decoder from DEM.
        decoder = select_decoder(dem, preferred=decoder_name, code=self.code)
        logger.debug("Decoder type: %s", type(decoder).__name__)

        # 5) Sample from the DEM directly.
        sampler = dem.compile_sampler()
        raw = sampler.sample(shots=shots)

        # Handle both possible APIs:
        #   * tuple of (det, obs, extra)
        #   * single array with det+obs concatenated
        if isinstance(raw, tuple):
            if len(raw) < 2:
                raise ValueError(f"DEM sampler returned tuple of length {len(raw)}; expected >= 2.")
            det_samples = np.asarray(raw[0], dtype=np.uint8)
            obs_samples = np.asarray(raw[1], dtype=np.uint8) if raw[1] is not None else None
        else:
            arr = np.asarray(raw, dtype=np.uint8)
            if arr.ndim != 2:
                raise ValueError(f"DEM sampler returned array with ndim={arr.ndim}; expected 2.")
            num_det = dem.num_detectors
            num_obs = dem.num_observables
            if arr.shape[1] != num_det + num_obs:
                raise ValueError(
                    f"DEM sampler array has shape {arr.shape}, but DEM has "
                    f"{num_det} detectors and {num_obs} observables."
                )
            det_samples = arr[:, :num_det]
            obs_samples = arr[:, num_det:]

        logger.debug("Samples: det=%s, obs=%s", det_samples.shape, obs_samples.shape)

        if det_samples is None or obs_samples is None:
            raise ValueError("DEM sampler did not return detector and observable samples as expected.")

        if det_samples.shape[0] != shots:
            raise ValueError(
                f"det_samples has {det_samples.shape[0]} shots, expected {shots}."
            )
        if obs_samples.shape[0] != shots:
            raise ValueError(
                f"obs_samples has {obs_samples.shape[0]} shots, expected {shots}."
            )

        # 6) Decode detector outcomes -> predicted logical flips.
        corrections = decoder.decode_batch(det_samples)
        corrections = np.asarray(corrections, dtype=np.uint8)

        # Ensure shape is (shots, num_observables)
        if corrections.ndim == 1:
            corrections = corrections.reshape(-1, dem.num_observables)
        if corrections.shape[0] != shots:
            raise ValueError(
                f"Decoder returned {corrections.shape[0]} shots but we asked for {shots}."
            )

        # 7) Compare predicted vs true logical bits for the first logical observable (L0).
        true_log = obs_samples[:, 0]           # shape (shots,)
        pred_log = corrections[:, 0]           # shape (shots,)

        logical_errors = (pred_log ^ true_log).astype(np.uint8)

        logger.debug("Logical error rate: %.6f", float(logical_errors.mean()))

        return {
            "shots": shots,
            "logical_errors": logical_errors,
            "logical_error_rate": float(logical_errors.mean()),
        }
    

    @staticmethod
    def run_decode_on_circuit(
        circuit: stim.Circuit,
        shots: int = 10_000,
        decoder_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        DEM + decoder-based logical error estimator.

        This version is compatible with the polyfill stim.Circuit and stim 1.15.0.

        Strategy:
          1. Build the (possibly noisy) circuit.
          2. Extract a DetectorErrorModel (DEM).
          3. Build a decoder from the DEM (e.g. PyMatchingDecoder).
          4. Sample *directly* from the DEM:
               - detector outcomes
               - logical observable bits
          5. Run the decoder on detector outcomes to get predicted logical flips.
          6. Compare predicted vs. true logical bits to estimate logical error rate.
        """
        print("[run_decode] --- starting ---")
        print("[run_decode] shots           =", shots)
        print("[run_decode] decoder_name    =", decoder_name)

        # print("[run_decode] base circuit summary:\n", str(base_circuit))

        print("[run_decode] noisy circuit   =", len(circuit), "instructions")

        # 3) Build DetectorErrorModel from noisy circuit.
        dem = circuit.detector_error_model(decompose_errors=True)
        print("[run_decode] DEM: detectors   =", dem.num_detectors)
        print("[run_decode] DEM: errors      =", dem.num_errors)
        print("[run_decode] DEM: observables =", dem.num_observables)
        print("[run_decode] DEM snippet:\n", str(dem)[:300])

        if dem.num_observables == 0:
            print("[run_decode] WARNING: DEM has no observables; returning zero logical errors.")
            logical_errors = np.zeros(shots, dtype=np.uint8)
            return {
                "shots": shots,
                "logical_errors": logical_errors,
                "logical_error_rate": float(logical_errors.mean()),
            }

        # 4) Build decoder from DEM.
        decoder = select_decoder(dem, preferred=decoder_name)
        print("[run_decode] decoder type    =", type(decoder))

        # 5) Sample from the DEM directly.
        #
        # For stim 1.15.0 + polyfill, dem.compile_sampler().sample(shots)
        # returns a 3-tuple:
        #   (det_samples, obs_samples, _)
        #
        # where:
        #   det_samples: (shots, num_detectors) bool
        #   obs_samples: (shots, num_observables) bool
        print("[run_decode] sampling DEM directly...")
        sampler = dem.compile_sampler()
        raw = sampler.sample(shots=shots)
        print("[run_decode] type(raw)       =", type(raw))

        # Handle both possible APIs:
        #   * tuple of (det, obs, extra)
        #   * single array with det+obs concatenated (older/newer variants)
        if isinstance(raw, tuple):
            if len(raw) < 2:
                raise ValueError(f"DEM sampler returned tuple of length {len(raw)}; expected >= 2.")
            det_samples = np.asarray(raw[0], dtype=np.uint8)
            obs_samples = np.asarray(raw[1], dtype=np.uint8) if raw[1] is not None else None
        else:
            # Assume raw is a 2D array of shape (shots, num_detectors + num_observables)
            arr = np.asarray(raw, dtype=np.uint8)
            if arr.ndim != 2:
                raise ValueError(f"DEM sampler returned array with ndim={arr.ndim}; expected 2.")
            num_det = dem.num_detectors
            num_obs = dem.num_observables
            if arr.shape[1] != num_det + num_obs:
                raise ValueError(
                    f"DEM sampler array has shape {arr.shape}, but DEM has "
                    f"{num_det} detectors and {num_obs} observables."
                )
            det_samples = arr[:, :num_det]
            obs_samples = arr[:, num_det:]

        print("[run_decode] det_samples.shape =", None if det_samples is None else det_samples.shape)
        print("[run_decode] obs_samples.shape =", None if obs_samples is None else obs_samples.shape)

        if det_samples is None or obs_samples is None:
            raise ValueError("DEM sampler did not return detector and observable samples as expected.")

        if det_samples.shape[0] != shots:
            raise ValueError(
                f"det_samples has {det_samples.shape[0]} shots, expected {shots}."
            )
        if obs_samples.shape[0] != shots:
            raise ValueError(
                f"obs_samples has {obs_samples.shape[0]} shots, expected {shots}."
            )

        print("[run_decode] first 5 det rows:\n", det_samples[:5])
        print("[run_decode] first 5 obs rows:\n", obs_samples[:5])

        # 6) Decode detector outcomes -> predicted logical flips.
        #    PyMatchingDecoder exposes decode_batch(det_samples) -> (shots, num_observables)
        print("[run_decode] decoding detector samples...")
        corrections = decoder.decode_batch(det_samples)
        corrections = np.asarray(corrections, dtype=np.uint8)
        print("[run_decode] corrections.shape =", corrections.shape)
        print("[run_decode] first 5 corrections rows:\n", corrections[:5])

        # Ensure shape is (shots, num_observables)
        if corrections.ndim == 1:
            corrections = corrections.reshape(-1, dem.num_observables)
        if corrections.shape[0] != shots:
            raise ValueError(
                f"Decoder returned {corrections.shape[0]} shots but we asked for {shots}."
            )
        if corrections.shape[1] != dem.num_observables:
            print(
                "[run_decode] WARNING: decoder returned",
                corrections.shape[1],
                "observables, but DEM has",
                dem.num_observables,
            )

        # 7) Compare predicted vs true logical bits for the first logical observable (L0).
        true_log = obs_samples[:, 0]           # shape (shots,)
        pred_log = corrections[:, 0]           # shape (shots,)

        logical_errors = (pred_log ^ true_log).astype(np.uint8)

        print("[run_decode] first 20 true_log:      ", true_log[:20])
        print("[run_decode] first 20 pred_log:      ", pred_log[:20])
        print("[run_decode] first 20 logical_errors:", logical_errors[:20])
        print("[run_decode] logical_error_rate =", float(logical_errors.mean()))

        return {
            "shots": shots,
            "logical_errors": logical_errors,
            "logical_error_rate": float(logical_errors.mean()),
        }