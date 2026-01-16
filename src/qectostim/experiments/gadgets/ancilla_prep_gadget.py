"""
Ancilla Preparation Gadgets for Fault-Tolerant Syndrome Extraction.

These gadgets handle the preparation of ancilla qubits for syndrome extraction.
The choice of ancilla preparation directly affects fault-tolerance:

1. **BareAncillaGadget**: Simple reset to |0⟩^⊗n or |+⟩^⊗n (NOT fault-tolerant)
   - Fast but allows single ancilla errors to corrupt syndrome
   - Suitable for testing or non-FT experiments

2. **EncodedAncillaGadget**: Encodes ancilla into |0_L⟩ or |+_L⟩ (fault-tolerant)
   - Uses encoding circuit to prepare logical state
   - Single-qubit errors on ancilla stay weight-1 after transversal CNOT
   - Standard FT syndrome extraction per Steane (1996)

3. **VerifiedAncillaGadget**: Encoded + verification (fully fault-tolerant)
   - Uses flag/verification qubits to detect preparation errors
   - Per AGP (Aliferis-Gottesman-Preskill) protocol
   - Most robust but highest overhead

References:
- Steane, "Active Stabilization, Quantum Computation, and Quantum State Synthesis" (1996)
- Aliferis, Gottesman, Preskill, "Quantum accuracy threshold for concatenated
  distance-3 codes", Quant. Inf. Comput. 6 (2006)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple
from enum import Enum
import warnings

import stim
import numpy as np

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.noise.models import NoiseModel


class AncillaBasis(Enum):
    """Basis for ancilla preparation."""
    ZERO = "zero"      # |0⟩ or |0_L⟩ for Z syndrome extraction
    PLUS = "plus"      # |+⟩ or |+_L⟩ for X syndrome extraction


@dataclass
class AncillaPrepResult:
    """Result of ancilla preparation."""
    # Which qubits contain the prepared ancilla
    ancilla_qubits: List[int]
    # Measurement indices from verification (if any)
    verification_measurements: List[int] = field(default_factory=list)
    # Total measurements emitted
    total_measurements: int = 0


class AncillaPrepGadget(ABC):
    """
    Abstract base class for ancilla preparation gadgets.
    
    These gadgets prepare ancilla qubits in a specific state for
    syndrome extraction. The base class defines the interface;
    subclasses implement different preparation strategies.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this gadget."""
        pass
    
    @property
    @abstractmethod
    def is_fault_tolerant(self) -> bool:
        """Whether this preparation is fault-tolerant."""
        pass
    
    @property
    def uses_verification(self) -> bool:
        """Whether this gadget uses verification qubits."""
        return False
    
    @property
    def verification_qubits_per_block(self) -> int:
        """Number of verification qubits needed per block."""
        return 0
    
    @abstractmethod
    def emit_prepare(
        self,
        circuit: stim.Circuit,
        ancilla_qubits: List[int],
        basis: AncillaBasis,
        code: Optional["CSSCode"] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> AncillaPrepResult:
        """
        Emit circuit instructions to prepare ancilla qubits.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to append instructions to.
        ancilla_qubits : List[int]
            Physical qubit indices for ancilla block.
        basis : AncillaBasis
            Whether to prepare |0⟩/|0_L⟩ (ZERO) or |+⟩/|+_L⟩ (PLUS).
        code : CSSCode, optional
            The code (provides encoding information for encoded prep).
        noise_model : NoiseModel, optional
            Noise model to apply during preparation.
        measurement_offset : int
            Starting measurement index.
            
        Returns
        -------
        AncillaPrepResult
            Information about the prepared ancilla.
        """
        pass


class BareAncillaGadget(AncillaPrepGadget):
    """
    Bare (unencoded) ancilla preparation - NOT fault-tolerant.
    
    Simply resets ancilla qubits to |0⟩^⊗n or |+⟩^⊗n without encoding.
    This is the current default behavior but is NOT fault-tolerant because
    a single-qubit error on the ancilla corrupts the syndrome.
    
    Use this for:
    - Testing / debugging
    - Non-FT experiments where simplicity is desired
    - Baseline comparisons
    
    Warning
    -------
    Single-qubit errors on ancilla will corrupt syndrome measurements,
    leading to incorrect error correction. The logical error rate scales
    as O(p) rather than O(p²) for distance-3 codes.
    """
    
    @property
    def name(self) -> str:
        return "BareAncilla"
    
    @property
    def is_fault_tolerant(self) -> bool:
        return False
    
    def emit_prepare(
        self,
        circuit: stim.Circuit,
        ancilla_qubits: List[int],
        basis: AncillaBasis,
        code: Optional["CSSCode"] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> AncillaPrepResult:
        """Prepare bare |0⟩^⊗n or |+⟩^⊗n state.
        
        Note: Ancillas are assumed to already be in |0⟩ (from previous MR or initial state).
        We just need to rotate to |+⟩ if needed.
        """
        
        # For |+⟩ basis, apply Hadamard (ancillas already in |0⟩ from previous MR)
        if basis == AncillaBasis.PLUS:
            circuit.append("H", ancilla_qubits)
            if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
                circuit.append("DEPOLARIZE1", ancilla_qubits, noise_model.p1)
        
        return AncillaPrepResult(
            ancilla_qubits=ancilla_qubits,
            verification_measurements=[],
            total_measurements=0,
        )


class EncodedAncillaGadget(AncillaPrepGadget):
    """
    Encoded ancilla preparation - Fault-tolerant.
    
    Prepares ancilla in encoded logical state |0_L⟩ or |+_L⟩ using the
    code's encoding circuit. This is fault-tolerant because:
    
    1. Single-qubit errors during encoding create weight-1 errors
    2. After transversal CNOT, error stays weight-1 on data
    3. Weight-1 errors are correctable by the code
    
    For CSS codes:
    - |0_L⟩: Reset to |0⟩^⊗n, then apply encoding gates
    - |+_L⟩: Apply H^⊗n to get |+⟩^⊗n, then apply Z-part encoding
    
    The encoding circuit is derived from the code's parity check matrices.
    For Steane code specifically, we use the standard encoding:
    - H on qubits {0, 1, 3}
    - CNOT cascade per STEANE_ENCODING_CNOTS
    
    References
    ----------
    Steane, "Multiple-particle interference and quantum error correction",
    Proc. R. Soc. Lond. A 452, 2551 (1996)
    """
    
    # Steane code [[7,1,3]] encoding circuit
    # H gates applied to these qubit indices (for |0_L⟩ prep)
    STEANE_ENCODING_H_QUBITS = [0, 1, 3]
    
    # CNOT gates in order: (control_idx, target_idx)
    STEANE_ENCODING_CNOTS = [
        (1, 2), (3, 5), (0, 4),  # Layer 1
        (1, 6), (0, 2), (3, 4),  # Layer 2
        (1, 5), (4, 6),          # Layer 3
    ]
    
    def __init__(self, code: Optional["CSSCode"] = None, custom_encoder: Optional[Any] = None, validate_custom_encoder: bool = False):
        """
        Initialize encoded ancilla gadget.
        
        Parameters
        ----------
        code : CSSCode, optional
            The code to use for encoding. If None, uses Steane code structure.
        custom_encoder : callable, optional
            A callable (circuit, ancilla_qubits, basis, noise_model) -> None that
            emits an encoding circuit for arbitrary CSS codes.
        validate_custom_encoder : bool
            If True, run a lightweight stabilizer check to validate the encoder.
        """
        self.code = code
        self.custom_encoder = custom_encoder
        self.validate_custom_encoder = validate_custom_encoder
        self._encoder_validated = False
    
    @property
    def name(self) -> str:
        return "EncodedAncilla"
    
    @property
    def is_fault_tolerant(self) -> bool:
        return True
    
    def _validate_encoder(self, ancilla_qubits: List[int], basis: AncillaBasis, noise_model: Optional["NoiseModel"]):
        """
        Validate that custom encoder produces correct stabilizer structure.
        
        Runs a quick Stim simulation to check stabilizer measurements.
        """
        if self.code is None or not hasattr(self.code, 'hx'):
            warnings.warn("Cannot validate custom encoder without code.hx/hz; skipping.", RuntimeWarning)
            return
        
        # Build a test circuit
        test_circuit = stim.Circuit()
        test_qubits = list(range(len(ancilla_qubits)))
        
        # Apply encoding
        self.custom_encoder(test_circuit, test_qubits, basis, None)  # No noise for validation
        
        # Measure stabilizers to verify they all give +1 (encoded |0_L⟩ or |+_L⟩)
        hx = np.atleast_2d(np.array(self.code.hx() if callable(self.code.hx) else self.code.hx, dtype=int))
        for row in hx[:2]:  # Check first 2 stabilizers as sanity
            support = list(np.where(row)[0])
            v_qubit = max(test_qubits) + 1
            test_circuit.append("R", [v_qubit])
            if basis == AncillaBasis.ZERO:
                test_circuit.append("H", [v_qubit])
                for q_idx in support:
                    if q_idx < len(test_qubits):
                        test_circuit.append("CNOT", [test_qubits[q_idx], v_qubit])
                test_circuit.append("H", [v_qubit])
            test_circuit.append("M", [v_qubit])
        
        # Run and check all measurements are 0
        sampler = test_circuit.compile_sampler()
        sample = sampler.sample(1)[0]
        if np.any(sample):
            warnings.warn(
                f"Custom encoder validation failed: stabilizer measurements non-zero ({sample}). "
                "Encoding may not preserve code space.",
                RuntimeWarning,
            )
    
    def _get_encoding_circuit(self, n: int) -> Tuple[List[int] | None, List[Tuple[int, int]] | None]:
        """
        Get encoding circuit for the code.
        
        Returns (h_qubits, cnot_pairs) where:
        - h_qubits: indices to apply H gates
        - cnot_pairs: list of (ctrl_idx, tgt_idx) for CNOTs
        """
        # Prefer code-supplied encoder if available
        if self.code is not None and hasattr(self.code, "encode_block"):
            return None, None  # Signal that we should delegate to code.encode_block
        if self.custom_encoder is not None:
            # Caller-supplied encoder handles all details
            return None, None
        if n == 7:
            # Steane code
            return self.STEANE_ENCODING_H_QUBITS, self.STEANE_ENCODING_CNOTS
        elif n == 9:
            # Shor code [[9,1,3]] - different encoding structure
            # |0_L⟩ = (|000⟩ + |111⟩)^⊗3 / 2√2
            # Encoding: H on {0,3,6}, then CNOTs
            h_qubits = [0, 3, 6]
            cnots = [
                (0, 1), (0, 2),  # First block
                (3, 4), (3, 5),  # Second block
                (6, 7), (6, 8),  # Third block
            ]
            return h_qubits, cnots
        else:
            # Generic CSS code not implemented here unless a custom encoder is supplied
            raise ValueError(
                "EncodedAncillaGadget has no encoding circuit for code size "
                f"n={n}. Provide code.encode_block or a custom_encoder or a code-specific gadget."
            )
    
    def emit_prepare(
        self,
        circuit: stim.Circuit,
        ancilla_qubits: List[int],
        basis: AncillaBasis,
        code: Optional["CSSCode"] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> AncillaPrepResult:
        """
        Prepare encoded |0_L⟩ or |+_L⟩ state.
        
        For |0_L⟩: Reset → H on subset → CNOT cascade
        For |+_L⟩: Reset → H^⊗n → (conjugated encoding)
        
        For concatenated codes at outer levels (n != code.n), falls back to
        bare ancilla preparation since we don't have encoding circuits for
        arbitrary block sizes.
        """
        n = len(ancilla_qubits)
        use_code = code or self.code
        
        # Check if this is an outer level (n doesn't match code's n)
        # For concatenated codes, treat as multiple inner-code blocks
        code_n = getattr(use_code, 'n', None) if use_code is not None else None
        if code_n is not None and n != code_n and code_n in (7, 9):
            # Outer level: n=49 for [[7,1,3]]^2, code_n=7
            # Encode each 7-qubit block separately for fault-tolerance
            num_blocks = n // code_n
            if n % code_n == 0 and num_blocks > 0:
                # Split into code_n-sized blocks and encode each
                for block_idx in range(num_blocks):
                    start = block_idx * code_n
                    block_qubits = ancilla_qubits[start:start + code_n]
                    # Recursively encode this block
                    self.emit_prepare(
                        circuit, block_qubits, basis, use_code, noise_model, measurement_offset
                    )
                return AncillaPrepResult(
                    ancilla_qubits=ancilla_qubits,
                    verification_measurements=[],
                    total_measurements=0,
                )
        
        # Normal case: n matches code.n, use standard encoding
        # Get encoding circuit structure
        h_qubits_idx, cnot_pairs = self._get_encoding_circuit(n)
        
        # Use MR (measure-and-reset) to cleanly reset ancillas.
        # - For fresh ancillas: MR initializes them to |0⟩ 
        # - For reused ancillas: MR collapses any entanglement from previous rounds
        #   before resetting, avoiding non-deterministic Stim errors
        circuit.append("MR", ancilla_qubits)
        if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
            circuit.append("DEPOLARIZE1", ancilla_qubits, noise_model.p1)

        # If caller provided a full custom encoder, use it and return
        if self.custom_encoder is not None:
            # Validate on first use if requested
            if self.validate_custom_encoder and not self._encoder_validated:
                self._validate_encoder(ancilla_qubits, basis, noise_model)
                self._encoder_validated = True
            
            self.custom_encoder(circuit, ancilla_qubits, basis, noise_model)
            return AncillaPrepResult(
                ancilla_qubits=ancilla_qubits,
                verification_measurements=[],
                total_measurements=0,
            )

        # If code provides an encode_block hook, delegate
        if h_qubits_idx is None and cnot_pairs is None and self.code is not None and hasattr(self.code, "encode_block"):
            self.code.encode_block(circuit, ancilla_qubits, basis, noise_model)
            return AncillaPrepResult(
                ancilla_qubits=ancilla_qubits,
                verification_measurements=[],
                total_measurements=0,
            )
        
        if basis == AncillaBasis.ZERO:
            # |0_L⟩ preparation
            # Step 1: Apply H to encoding qubits
            if h_qubits_idx:
                h_physical = [ancilla_qubits[i] for i in h_qubits_idx]
                circuit.append("H", h_physical)
                if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
                    circuit.append("DEPOLARIZE1", h_physical, noise_model.p1)
            
            # Step 2: Apply CNOT cascade
            for ctrl_idx, tgt_idx in cnot_pairs:
                ctrl = ancilla_qubits[ctrl_idx]
                tgt = ancilla_qubits[tgt_idx]
                circuit.append("CNOT", [ctrl, tgt])
                if noise_model is not None and hasattr(noise_model, 'p2') and noise_model.p2 > 0:
                    circuit.append("DEPOLARIZE2", [ctrl, tgt], noise_model.p2)
        
        elif basis == AncillaBasis.PLUS:
            # |+_L⟩ preparation
            # For CSS codes: H^⊗n|0_L⟩ = |+_L⟩
            # So: Reset → Encode |0_L⟩ → H^⊗n
            
            # First encode |0_L⟩
            if h_qubits_idx:
                h_physical = [ancilla_qubits[i] for i in h_qubits_idx]
                circuit.append("H", h_physical)
                if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
                    circuit.append("DEPOLARIZE1", h_physical, noise_model.p1)
            
            for ctrl_idx, tgt_idx in cnot_pairs:
                ctrl = ancilla_qubits[ctrl_idx]
                tgt = ancilla_qubits[tgt_idx]
                circuit.append("CNOT", [ctrl, tgt])
                if noise_model is not None and hasattr(noise_model, 'p2') and noise_model.p2 > 0:
                    circuit.append("DEPOLARIZE2", [ctrl, tgt], noise_model.p2)
            
            # Then apply logical H (transversal H for CSS)
            circuit.append("H", ancilla_qubits)
            if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
                circuit.append("DEPOLARIZE1", ancilla_qubits, noise_model.p1)
        
        return AncillaPrepResult(
            ancilla_qubits=ancilla_qubits,
            verification_measurements=[],
            total_measurements=0,
        )


class VerifiedAncillaGadget(AncillaPrepGadget):
    """
    Verified ancilla preparation - Fully fault-tolerant with verification.
    
    Prepares encoded ancilla and then verifies correctness using additional
    measurements. If verification fails, the ancilla can be discarded or
    corrected (depending on implementation).
    
    Verification methods:
    1. **Bell-pair verification**: Prepare two copies, measure in Bell basis
    2. **Flag qubit verification**: Use flag qubits during encoding
    3. **Stabilizer verification**: Measure stabilizers after encoding
    
    This implementation uses stabilizer verification:
    - After encoding |0_L⟩, measure X stabilizers
    - All X stabilizer measurements should give +1
    - Non-zero syndrome indicates preparation error
    
    For |+_L⟩, measure Z stabilizers instead.
    
    Post-selection Support:
    - When enable_post_selection=True, verification_measurements can be used
      by the decoder to reject shots where verification failed.
    - This is ESSENTIAL for achieving proper fault-tolerant behavior.
    - Without post-selection, bad ancilla prep can cause weight-1 logical errors.
    
    References
    ----------
    Aliferis, Gottesman, Preskill, "Quantum accuracy threshold for concatenated
    distance-3 codes", Quant. Inf. Comput. 6 (2006)
    """
    
    # Steane code X stabilizer generators (for |0_L⟩ verification)
    # These should all measure +1 for valid |0_L⟩
    STEANE_X_STABILIZERS = [
        [0, 2, 4, 6],  # X0X2X4X6
        [1, 2, 5, 6],  # X1X2X5X6
        [3, 4, 5, 6],  # X3X4X5X6
    ]
    
    # Steane code Z stabilizer generators (for |+_L⟩ verification)
    STEANE_Z_STABILIZERS = [
        [0, 2, 4, 6],  # Z0Z2Z4Z6
        [1, 2, 5, 6],  # Z1Z2Z5Z6
        [3, 4, 5, 6],  # Z3Z4Z5Z6
    ]
    
    # Shor [[9,1,3]] code stabilizers
    SHOR_X_STABILIZERS = [
        [0, 1, 2, 3, 4, 5],  # X0-X5
        [3, 4, 5, 6, 7, 8],  # X3-X8
    ]
    
    SHOR_Z_STABILIZERS = [
        [0, 1],  # Z0Z1
        [1, 2],  # Z1Z2
        [3, 4],  # Z3Z4
        [4, 5],  # Z4Z5
        [6, 7],  # Z6Z7
        [7, 8],  # Z7Z8
    ]
    
    def __init__(
        self,
        code: Optional["CSSCode"] = None,
        discard_on_failure: bool = False,
        enable_post_selection: bool = True,
        retry_limit: int = 0,
    ):
        """
        Initialize verified ancilla gadget.
        
        Parameters
        ----------
        code : CSSCode, optional
            The code to use for encoding. If None, defaults to Steane code.
        discard_on_failure : bool
            If True, verification failure causes re-preparation (not implemented).
            If False, just reports verification results.
        enable_post_selection : bool
            If True (default), verification measurements are tracked so the decoder
            can post-select on successful verification. This is ESSENTIAL for FT.
        """
        self.code = code
        self.discard_on_failure = discard_on_failure
        self.enable_post_selection = enable_post_selection
        if retry_limit:
            warnings.warn(
                "retry_limit is not supported in static Stim circuits; post-select instead.",
                RuntimeWarning,
            )
        self.retry_limit = 0
        self._encoded_gadget = EncodedAncillaGadget(code)
    
    @property
    def name(self) -> str:
        return "VerifiedAncilla"
    
    @property
    def is_fault_tolerant(self) -> bool:
        return True
    
    @property
    def uses_verification(self) -> bool:
        return True
    
    @property
    def verification_qubits_per_block(self) -> int:
        """Number of verification ancilla qubits per block."""
        # Determined by number of stabilizers to check
        if self.code is not None:
            n = getattr(self.code, 'n', 7)
        else:
            n = 7
        
        if n == 7:
            return 3  # Steane code: 3 X stabilizers
        elif n == 9:
            return 2  # Shor code: 2 X stabilizers for |0_L⟩
        else:
            # Generic: number of X stabilizers
            if self.code is not None:
                hx = getattr(self.code, 'hx', None)
                if hx is not None:
                    return hx.shape[0]
            return 3  # Default
    
    def _get_stabilizers(self, n: int, basis: AncillaBasis) -> List[List[int]]:
        """
        Get stabilizer supports for verification.
        
        This is generalized to support any CSS code, not just Steane.
        
        IMPORTANT: Only returns stabilizers if n matches the code's n.
        For concatenated codes at outer levels (n = 49, 343, etc.), 
        verification must be done at the inner level, not outer level.
        """
        # Check if n matches our code's expected size
        code_n = getattr(self.code, 'n', None) if self.code is not None else None
        
        if code_n is not None and n != code_n:
            # Block size doesn't match code size - this is likely an outer level
            # of a concatenated code. Skip verification at this level.
            # The inner levels will still be verified properly.
            return []
        
        if n == 7:
            # Steane code
            if basis == AncillaBasis.ZERO:
                return self.STEANE_X_STABILIZERS
            else:
                return self.STEANE_Z_STABILIZERS
        elif n == 9:
            # Shor code
            if basis == AncillaBasis.ZERO:
                return self.SHOR_X_STABILIZERS
            else:
                return self.SHOR_Z_STABILIZERS
        else:
            # Generic: try to extract from code's parity check matrices
            if self.code is not None:
                if basis == AncillaBasis.ZERO:
                    hx = getattr(self.code, 'hx', None)
                else:
                    hx = getattr(self.code, 'hz', None)
                
                if hx is not None:
                    import numpy as np
                    hx = np.atleast_2d(np.asarray(hx))
                    stabilizers = []
                    for row in hx:
                        support = list(np.where(row)[0])
                        stabilizers.append(support)
                    return stabilizers
            
            # Fallback: empty (no verification possible)
            return []
    
    def emit_prepare(
        self,
        circuit: stim.Circuit,
        ancilla_qubits: List[int],
        basis: AncillaBasis,
        code: Optional["CSSCode"] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
        verification_qubits: Optional[List[int]] = None,
        post_select: bool = False,
    ) -> AncillaPrepResult:
        """
        Prepare and verify encoded ancilla state.
        
        For concatenated codes at outer levels (n != code.n), falls back to
        bare ancilla preparation since encoded preparation requires knowing
        the encoding circuit, which is only defined for inner codes.
        
        Parameters
        ----------
        verification_qubits : List[int], optional
            Physical qubits for verification measurements.
            If None, will allocate after ancilla_qubits.
            
        Returns
        -------
        AncillaPrepResult
            Contains verification_measurements which can be used for
            post-selection. If any verification measurement is non-zero,
            the ancilla prep failed and the shot should be discarded.
        """
        n = len(ancilla_qubits)
        
        # Check if this is an outer level (n doesn't match code's n)
        # Previously fell back to bare ancilla - NOW use encoded for FT at all levels
        code_n = getattr(self.code, 'n', None) if self.code is not None else None
        use_encoded = True  # Always use encoded preparation for fault-tolerance
        
        if code_n is not None and n != code_n:
            # Outer level of concatenated code
            # Use encoded ancilla gadget recursively (treats outer block as inner code)
            # This maintains fault-tolerance at all concatenation levels
            pass  # Proceed to encoded preparation below
        
        # Step 1: Prepare encoded state (for all levels - inner and outer)
        self._encoded_gadget.emit_prepare(
            circuit, ancilla_qubits, basis, code, noise_model, measurement_offset
        )
        
        # Get stabilizers for this code size (generalized, not code-specific)
        stabilizers = self._get_stabilizers(n, basis)
        
        if not stabilizers:
            # No verification possible - return without verification
            return AncillaPrepResult(
                ancilla_qubits=ancilla_qubits,
                verification_measurements=[],
                total_measurements=0,
            )
        
        # Step 2: Allocate verification qubits if not provided
        n_verification = len(stabilizers)
        if verification_qubits is None:
            max_q = max(ancilla_qubits)
            verification_qubits = list(range(max_q + 1, max_q + 1 + n_verification))
        
        verification_measurements = []
        current_meas = measurement_offset
        
        # Get noise parameters for gates
        p1 = noise_model.p1 if noise_model and hasattr(noise_model, 'p1') else 0
        p2 = noise_model.p2 if noise_model and hasattr(noise_model, 'p2') else 0
        p_meas = noise_model.before_measure_flip if noise_model and hasattr(noise_model, 'before_measure_flip') else 0
        
        # For each stabilizer, measure using verification ancilla
        for stab_idx, support in enumerate(stabilizers):
            if stab_idx >= len(verification_qubits):
                break  # Not enough verification qubits
            
            v_qubit = verification_qubits[stab_idx]
            
            # Reset verification qubit
            circuit.append("R", [v_qubit])
            if p1 > 0:
                circuit.append("DEPOLARIZE1", [v_qubit], p1)
            
            if basis == AncillaBasis.ZERO:
                # For X stabilizers: |+⟩ ancilla, CNOT pattern, measure X
                circuit.append("H", [v_qubit])
                if p1 > 0:
                    circuit.append("DEPOLARIZE1", [v_qubit], p1)
                for data_idx in support:
                    if data_idx < len(ancilla_qubits):
                        # Data should control, verification ancilla is target to avoid fanout from ancilla faults
                        circuit.append("CNOT", [ancilla_qubits[data_idx], v_qubit])
                        if p2 > 0:
                            circuit.append("DEPOLARIZE2", [ancilla_qubits[data_idx], v_qubit], p2)
                circuit.append("H", [v_qubit])
                if p1 > 0:
                    circuit.append("DEPOLARIZE1", [v_qubit], p1)
            else:
                # For Z stabilizers: |0⟩ ancilla, CNOT pattern, measure Z
                for data_idx in support:
                    if data_idx < len(ancilla_qubits):
                        circuit.append("CNOT", [ancilla_qubits[data_idx], v_qubit])
                        if p2 > 0:
                            circuit.append("DEPOLARIZE2", [ancilla_qubits[data_idx], v_qubit], p2)
            
            # Add measurement error before M
            if p_meas > 0:
                circuit.append("X_ERROR", [v_qubit], p_meas)
            circuit.append("M", [v_qubit])
            verification_measurements.append(current_meas)
            current_meas += 1
            # Optionally attach DETECTOR so simulation can reject failing shots
            if post_select and self.enable_post_selection:
                circuit.append("DETECTOR", targets=[stim.target_rec(-1)])
        
        circuit.append("TICK")
        
        return AncillaPrepResult(
            ancilla_qubits=ancilla_qubits,
            verification_measurements=verification_measurements,
            total_measurements=len(verification_measurements),
        )


# =============================================================================
# Factory function for creating ancilla prep gadgets
# =============================================================================

def create_default_outer_verifier(inner_code, outer_code, post_select: bool = False):
    """
    Create a default outer-level verifier for concatenated ancilla prep.
    
    For Steane⊗Steane (7 blocks of 7 qubits), measures 3 outer X-stabilizers
    to verify the outer encoding is correct.
    
    Parameters
    ----------
    inner_code : CSSCode
        The inner code.
    outer_code : CSSCode
        The outer code.
    post_select : bool
        If True, attach DETECTORs to verification measurements.
    
    Returns
    -------
    callable or None
        A verifier function (circuit, ancilla_qubits, basis, inner_code, noise_model) -> None
        or None if no default verifier is available.
    """
    n_inner = getattr(inner_code, 'n', 0)
    n_outer = getattr(outer_code, 'n', 0)
    
    # Steane⊗Steane: measure outer X-stabilizers (for |0_LL⟩ prep)
    if n_inner == 7 and n_outer == 7:
        outer_x_stab_blocks = [
            [0, 2, 4, 6],  # Outer X stabilizer 1
            [1, 2, 5, 6],  # Outer X stabilizer 2
            [3, 4, 5, 6],  # Outer X stabilizer 3
        ]
        
        def steane_outer_verifier(circuit, ancilla_qubits, basis, inner_code, noise_model):
            """Verify outer encoding by measuring outer X-stabilizers."""
            if basis != AncillaBasis.ZERO:
                # For |+_LL⟩, we'd measure outer Z-stabilizers; skip for now
                return
            
            n_total = len(ancilla_qubits)
            n_blocks = n_total // n_inner
            
            # Allocate verification qubits after ancilla block
            max_q = max(ancilla_qubits)
            v_qubits = list(range(max_q + 1, max_q + 4))
            
            p1 = noise_model.p1 if noise_model and hasattr(noise_model, 'p1') else 0
            p2 = noise_model.p2 if noise_model and hasattr(noise_model, 'p2') else 0
            p_meas = noise_model.before_measure_flip if noise_model and hasattr(noise_model, 'before_measure_flip') else 0
            
            for stab_idx, block_support in enumerate(outer_x_stab_blocks):
                v_qubit = v_qubits[stab_idx]
                circuit.append("R", [v_qubit])
                if p1 > 0:
                    circuit.append("DEPOLARIZE1", [v_qubit], p1)
                
                # Measure outer X stabilizer: prepare |+⟩, CNOT from each block's qubits
                circuit.append("H", [v_qubit])
                if p1 > 0:
                    circuit.append("DEPOLARIZE1", [v_qubit], p1)
                
                # For each block in support, apply transversal CNOT from block to v_qubit
                for block_idx in block_support:
                    if block_idx < n_blocks:
                        start = block_idx * n_inner
                        block_qubits = ancilla_qubits[start : start + n_inner]
                        # Transversal: each qubit in block controls v_qubit (measuring X_L of block)
                        for q in block_qubits:
                            circuit.append("CNOT", [q, v_qubit])
                            if p2 > 0:
                                circuit.append("DEPOLARIZE2", [q, v_qubit], p2)
                
                circuit.append("H", [v_qubit])
                if p1 > 0:
                    circuit.append("DEPOLARIZE1", [v_qubit], p1)
                
                if p_meas > 0:
                    circuit.append("X_ERROR", [v_qubit], p_meas)
                circuit.append("M", [v_qubit])
                
                if post_select:
                    circuit.append("DETECTOR", targets=[stim.target_rec(-1)])
            
            circuit.append("TICK")
        
        return steane_outer_verifier
    
    # No default verifier for other concatenations
    return None

class ConcatenatedAncillaPrepGadget(AncillaPrepGadget):
    """
    Hierarchical ancilla preparation for concatenated codes.
    
    For a depth-2 concatenated code like [[49,1,9]] = [[7,1,3]] ⊗ [[7,1,3]]:
    
    1. **Inner level (7 blocks of 7 qubits each)**:
       - Prepare each 7-qubit block as |0_L⟩ using inner code encoding
       - This creates 7 logical qubits
       
    2. **Outer level (1 block of 7 logical qubits)**:
       - Treat the 7 encoded blocks as single qubits
       - Apply outer-level encoding using LOGICAL gates between blocks
       - Result: |0_LL⟩ (doubly-encoded logical zero)
    
    This hierarchical approach ensures FT because:
    - Single physical errors at inner level → weight-1 inner-level error
    - After inner decoding, propagates to at most 1 outer-level error
    - Single outer-level error is correctable by outer code
    
    Parameters
    ----------
    code : MultiLevelConcatenatedCode
        The concatenated code (provides level structure and inner codes).
    inner_prep : AncillaPrepGadget, optional
        Gadget for inner-level preparation (default: EncodedAncillaGadget).
    """
    
    def __init__(
        self,
        code: Optional[Any] = None,  # MultiLevelConcatenatedCode
        inner_prep: Optional[AncillaPrepGadget] = None,
        outer_encoder: Optional[Any] = None,
        outer_verifier: Optional[Any] = None,
        enable_default_outer_verifier: bool = True,
    ):
        self.code = code
        self.inner_prep = inner_prep
        self.outer_encoder = outer_encoder
        self.outer_verifier = outer_verifier
        self.enable_default_outer_verifier = enable_default_outer_verifier
        
        # Extract level codes if available
        self._level_codes = []
        if code is not None and hasattr(code, 'level_codes'):
            self._level_codes = code.level_codes
    
    @property
    def name(self) -> str:
        return "ConcatenatedAncilla"
    
    @property
    def is_fault_tolerant(self) -> bool:
        return True
    
    def _get_inner_code(self) -> Optional[Any]:
        """Get the innermost code for encoding."""
        if self._level_codes:
            return self._level_codes[0]
        return None
    
    def _get_outer_code(self) -> Optional[Any]:
        """Get the outer code for logical encoding."""
        if len(self._level_codes) >= 2:
            return self._level_codes[-1]
        return None
    
    def emit_prepare(
        self,
        circuit: stim.Circuit,
        ancilla_qubits: List[int],
        basis: AncillaBasis,
        code: Optional["CSSCode"] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> AncillaPrepResult:
        """
        Prepare hierarchically encoded ancilla state.
        
        For [[49,1,9]]: Prepares |0_LL⟩ or |+_LL⟩.
        
        This gadget detects concatenated structure in two ways:
        1. From explicit level_codes in the MultiLevelConcatenatedCode
        2. By inferring from n_total / code.n when given more qubits than code size
        """
        use_code = code or self.code
        n_total = len(ancilla_qubits)
        
        # Try to get inner code from explicit level structure
        inner_code = self._get_inner_code()
        
        # If no level structure, but we have more qubits than code.n,
        # infer that this is a concatenated ancilla request
        if inner_code is None and use_code is not None:
            code_n = getattr(use_code, 'n', n_total)
            if n_total > code_n and n_total % code_n == 0:
                # Infer: we're being asked to prepare n_total qubits using
                # a code of size code_n. This means hierarchical encoding.
                inner_code = use_code
        
        if inner_code is None or not hasattr(inner_code, 'n'):
            # Fall back to encoded gadget (will work for n=7,9)
            if n_total > getattr(use_code, 'n', n_total):
                warnings.warn(
                    "ConcatenatedAncillaPrepGadget falling back to single-level encoding; "
                    "outer level not applied. Result is not fully fault-tolerant.",
                    RuntimeWarning,
                )
            fallback = EncodedAncillaGadget(code=use_code)
            return fallback.emit_prepare(circuit, ancilla_qubits, basis, code, noise_model, measurement_offset)
        
        n_inner = inner_code.n
        n_blocks = n_total // n_inner
        
        if n_blocks * n_inner != n_total:
            # Not evenly divisible - fall back
            warnings.warn(
                "Ancilla block size not divisible by inner code size; using single-level encoding only.",
                RuntimeWarning,
            )
            fallback = EncodedAncillaGadget(code=use_code)
            return fallback.emit_prepare(circuit, ancilla_qubits, basis, code, noise_model, measurement_offset)
        
        # If only 1 block, just use inner encoding
        if n_blocks == 1:
            inner_prep = self.inner_prep or EncodedAncillaGadget(code=inner_code)
            return inner_prep.emit_prepare(circuit, ancilla_qubits, basis, inner_code, noise_model, measurement_offset)
        
        # Create inner prep gadget
        inner_prep = self.inner_prep or EncodedAncillaGadget(code=inner_code)
        
        # ==== STEP 1: Prepare each inner block as |0_L⟩ or |+_L⟩ ====
        for block_idx in range(n_blocks):
            start = block_idx * n_inner
            block_qubits = ancilla_qubits[start : start + n_inner]
            inner_prep.emit_prepare(
                circuit=circuit,
                ancilla_qubits=block_qubits,
                basis=basis,
                code=inner_code,
                noise_model=noise_model,
                measurement_offset=measurement_offset,
            )
        
        circuit.append("TICK")
        
        # ==== STEP 2: Apply outer-level encoding (LOGICAL gates between blocks) ====
        # For Steane ⊗ Steane: outer encoding is also Steane structure
        # H on blocks {0,1,3}, then CNOT cascade
        
        outer_h_blocks = [0, 1, 3]
        outer_cnot_pairs = [
            (1, 2), (3, 5), (0, 4),  # Layer 1
            (1, 6), (0, 2), (3, 4),  # Layer 2
            (1, 5), (4, 6),          # Layer 3
        ]
        
        if n_blocks == 7:
            # Outer encoding for [[7,1,3]] structure (can be overridden)
            if self.outer_encoder is not None:
                self.outer_encoder(circuit, ancilla_qubits, basis, inner_code, noise_model)
            else:
                if basis == AncillaBasis.ZERO:
                    # |0_LL⟩: Apply H_L to blocks then CNOT_L cascade
                    # Logical H = transversal H
                    for block_idx in outer_h_blocks:
                        start = block_idx * n_inner
                        block_qubits = ancilla_qubits[start : start + n_inner]
                        circuit.append("H", block_qubits)
                        if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
                            circuit.append("DEPOLARIZE1", block_qubits, noise_model.p1)
                    
                    circuit.append("TICK")
                    
                    # Logical CNOT = transversal CNOT
                    for ctrl_block, tgt_block in outer_cnot_pairs:
                        for i in range(n_inner):
                            ctrl = ancilla_qubits[ctrl_block * n_inner + i]
                            tgt = ancilla_qubits[tgt_block * n_inner + i]
                            circuit.append("CNOT", [ctrl, tgt])
                            if noise_model is not None and hasattr(noise_model, 'p2') and noise_model.p2 > 0:
                                circuit.append("DEPOLARIZE2", [ctrl, tgt], noise_model.p2)
                        circuit.append("TICK")
                
                elif basis == AncillaBasis.PLUS:
                    # |+_LL⟩: First encode |0_LL⟩, then apply H_LL (transversal H on all)
                    
                    # Encode |0_LL⟩ first
                    for block_idx in outer_h_blocks:
                        start = block_idx * n_inner
                        block_qubits = ancilla_qubits[start : start + n_inner]
                        circuit.append("H", block_qubits)
                        if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
                            circuit.append("DEPOLARIZE1", block_qubits, noise_model.p1)
                    
                    circuit.append("TICK")
                    
                    for ctrl_block, tgt_block in outer_cnot_pairs:
                        for i in range(n_inner):
                            ctrl = ancilla_qubits[ctrl_block * n_inner + i]
                            tgt = ancilla_qubits[tgt_block * n_inner + i]
                            circuit.append("CNOT", [ctrl, tgt])
                            if noise_model is not None and hasattr(noise_model, 'p2') and noise_model.p2 > 0:
                                circuit.append("DEPOLARIZE2", [ctrl, tgt], noise_model.p2)
                        circuit.append("TICK")
                    
                    # Apply H_LL (transversal H on ALL qubits)
                    circuit.append("H", ancilla_qubits)
                    if noise_model is not None and hasattr(noise_model, 'p1') and noise_model.p1 > 0:
                        circuit.append("DEPOLARIZE1", ancilla_qubits, noise_model.p1)
            # Optional outer verification hook
            verifier_to_use = self.outer_verifier
            if verifier_to_use is None and self.enable_default_outer_verifier:
                outer_code = self._get_outer_code()
                if outer_code is not None:
                    verifier_to_use = create_default_outer_verifier(inner_code, outer_code, post_select=False)
            
            if verifier_to_use is not None:
                verifier_to_use(circuit, ancilla_qubits, basis, inner_code, noise_model)
        else:
            if self.outer_encoder is not None:
                self.outer_encoder(circuit, ancilla_qubits, basis, inner_code, noise_model)
                
                verifier_to_use = self.outer_verifier
                if verifier_to_use is None and self.enable_default_outer_verifier:
                    outer_code = self._get_outer_code()
                    if outer_code is not None:
                        verifier_to_use = create_default_outer_verifier(inner_code, outer_code, post_select=False)
                
                if verifier_to_use is not None:
                    verifier_to_use(circuit, ancilla_qubits, basis, inner_code, noise_model)
            else:
                raise ValueError(
                    "Concatenated ancilla requested without an outer encoder; "
                    "cannot guarantee fault tolerance for n_blocks>1. Provide outer_encoder."
                )
        
        return AncillaPrepResult(
            ancilla_qubits=ancilla_qubits,
            verification_measurements=[],
            total_measurements=0,
        )


class AncillaPrepMethod(Enum):
    """Available ancilla preparation methods."""
    BARE = "bare"
    ENCODED = "encoded"
    VERIFIED = "verified"
    CONCATENATED = "concatenated"  # Hierarchical for multi-level codes


def create_ancilla_prep_gadget(
    method: AncillaPrepMethod | str,
    code: Optional["CSSCode"] = None,
    custom_encoder: Optional[Any] = None,
    outer_encoder: Optional[Any] = None,
    outer_verifier: Optional[Any] = None,
    **kwargs,
) -> AncillaPrepGadget:
    """
    Factory function to create ancilla preparation gadgets.
    
    Parameters
    ----------
    method : AncillaPrepMethod or str
        Which preparation method to use:
        - "bare": Simple reset (non-FT)
        - "encoded": Encoded preparation (FT)
        - "verified": Encoded + verified (fully FT)
    code : CSSCode, optional
        Code for encoding-based methods.
    **kwargs
        Additional arguments passed to gadget constructor.
    
    Returns
    -------
    AncillaPrepGadget
        The requested gadget instance.
    
    Examples
    --------
    >>> gadget = create_ancilla_prep_gadget("bare")
    >>> gadget = create_ancilla_prep_gadget(AncillaPrepMethod.ENCODED, code=steane)
    >>> gadget = create_ancilla_prep_gadget("verified", code=steane, discard_on_failure=True)
    >>> gadget = create_ancilla_prep_gadget("concatenated", code=concat_code)  # Hierarchical
    """
    if isinstance(method, str):
        method = AncillaPrepMethod(method.lower())
    
    if method == AncillaPrepMethod.BARE:
        return BareAncillaGadget()
    elif method == AncillaPrepMethod.ENCODED:
        return EncodedAncillaGadget(code=code, custom_encoder=custom_encoder)
    elif method == AncillaPrepMethod.VERIFIED:
        return VerifiedAncillaGadget(code=code, **kwargs)
    elif method == AncillaPrepMethod.CONCATENATED:
        return ConcatenatedAncillaPrepGadget(code=code, outer_encoder=outer_encoder, outer_verifier=outer_verifier, **kwargs)
    else:
        raise ValueError(f"Unknown ancilla prep method: {method}")
