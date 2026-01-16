"""
No-Op Gadget: Simple idle noise, no error correction.

This is the simplest gadget - just applies idle noise to data qubits
without any syndrome extraction or correction. Used for simple memory
experiments matching the paper's approach.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import stim
import numpy as np

from .base import Gadget, MeasurementMap, SyndromeSchedule, LogicalMeasurementMap

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode
    from qectostim.noise.models import NoiseModel


class NoOpGadget(Gadget):
    """
    No-op gadget that just applies idle noise.
    
    This matches the paper's simple memory experiment:
    - No syndrome extraction
    - No error correction
    - Just idle noise during storage
    
    Used with:
    - SimpleConcatenatedMemoryExperiment
    - HardDecisionHierarchicalDecoder
    """
    
    def __init__(self, p_idle: float = 0.0):
        """
        Initialize no-op gadget.
        
        Parameters
        ----------
        p_idle : float
            Idle depolarizing error probability
        """
        self.p_idle = p_idle
    
    @property
    def name(self) -> str:
        return "NoOp"
    
    @property
    def requires_ancillas(self) -> bool:
        return False
    
    @property
    def ancillas_per_block(self) -> int:
        return 0
    
    def emit(
        self,
        circuit: stim.Circuit,
        data_qubits: Dict[int, List[int]],
        ancilla_qubits: Optional[Dict[int, List[int]]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
    ) -> MeasurementMap:
        """
        Emit idle noise on all data qubits.
        
        This gadget doesn't emit any measurements - it just applies
        noise during an "idle" period.
        """
        # Collect all data qubits
        all_data = []
        for block_id in sorted(data_qubits.keys()):
            all_data.extend(data_qubits[block_id])
        
        # Apply idle noise
        if self.p_idle > 0 and all_data:
            circuit.append("DEPOLARIZE1", all_data, self.p_idle)
        
        # No measurements from this gadget
        return MeasurementMap(
            offset=measurement_offset,
            total_measurements=0,
        )
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        """No syndromes measured."""
        return SyndromeSchedule()
    
    def get_logical_map(
        self,
        inner_code: "CSSCode",
        outer_code: "CSSCode",
    ) -> LogicalMeasurementMap:
        """
        Get logical measurement mapping.
        
        Since this gadget doesn't measure anything, the logical map
        is based on the final data measurements (emitted separately).
        """
        lmap = LogicalMeasurementMap()
        
        # Get inner Z_L support
        inner_z_support = _get_z_support(inner_code)
        for block_id in range(outer_code.n):
            lmap.inner_z_support[block_id] = inner_z_support
        
        # Get outer Z_L support
        lmap.outer_z_support = _get_z_support(outer_code)
        
        return lmap


class IdleGadget(NoOpGadget):
    """Alias for NoOpGadget."""
    
    @property
    def name(self) -> str:
        return "Idle"


def _get_z_support(code: "CSSCode") -> List[int]:
    """Helper to get Z logical operator support."""
    if hasattr(code, 'logical_z_support'):
        try:
            return list(code.logical_z_support(0))
        except:
            pass
    
    # Try parsing logical_z_ops
    if hasattr(code, 'logical_z_ops'):
        ops = code.logical_z_ops
        if callable(ops):
            ops = ops()
        if ops and len(ops) > 0:
            op = ops[0]
            if isinstance(op, str):
                return [i for i, c in enumerate(op) if c in ('Z', 'Y')]
    
    # Code-specific fallbacks
    code_name = getattr(code, 'name', '') or str(type(code).__name__)
    if 'shor' in code_name.lower() or 'Shor' in code_name:
        return list(range(code.n))  # All qubits for Shor
    if 'steane' in code_name.lower() or 'Stean' in code_name:
        return [0, 1, 2]  # Standard Steane Z_L
    
    # Generic fallback
    return list(range(min(3, code.n)))
