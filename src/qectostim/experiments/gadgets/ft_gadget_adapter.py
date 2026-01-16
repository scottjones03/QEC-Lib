# src/qectostim/experiments/gadgets/ft_gadget_adapter.py
"""
Fault-Tolerant Gadget Adapter - Bridges gadgets/ and experiments/gadgets/ systems.

This adapter allows using FT gate gadgets from gadgets/ within EC experiments:
- Wraps gadgets/base.Gadget to expose experiments/gadgets/base.Gadget interface
- Enables using TransversalHadamard, TransversalCNOT etc in multi-level experiments
- Handles the translation between PhaseResult and MeasurementMap

Use Cases:
1. Apply a logical gate within a memory experiment (e.g., H between EC rounds)
2. Use TransversalGadgets for syndrome extraction scheduling
3. Compose FT gates with EC gadgets in complex experiments

Example:
    >>> from qectostim.gadgets.transversal import TransversalHadamard
    >>> from qectostim.experiments.gadgets.ft_gadget_adapter import FTGadgetAdapter
    >>> 
    >>> ft_h = TransversalHadamard()
    >>> ec_h = FTGadgetAdapter(ft_h, code)
    >>> 
    >>> # Now usable in EC experiment context
    >>> mmap = ec_h.emit(circuit, level=0, block_idx=0)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

import stim
import numpy as np

from qectostim.experiments.gadgets.base import (
    Gadget as ECGadget,
    MeasurementMap,
    MeasurementEntry,
    SyndromeSchedule,
    LogicalMeasurementMap,
)

if TYPE_CHECKING:
    from qectostim.gadgets.base import Gadget as FTGadget, PhaseResult
    from qectostim.codes.abstract_code import Code
    from qectostim.noise.models import NoiseModel


@dataclass
class QubitMapping:
    """Mapping between local and global qubit indices."""
    data_qubits: List[int]
    x_ancilla_qubits: List[int] = field(default_factory=list)
    z_ancilla_qubits: List[int] = field(default_factory=list)
    
    @property
    def all_qubits(self) -> List[int]:
        return self.data_qubits + self.x_ancilla_qubits + self.z_ancilla_qubits


class FTGadgetAdapter(ECGadget):
    """
    Adapts an FT gate gadget (from gadgets/) to EC gadget interface.
    
    This enables using the sophisticated gadgets from gadgets/ (with
    phase-based emission, stabilizer transforms, etc.) within the
    simpler EC experiment framework.
    
    Parameters
    ----------
    ft_gadget : FTGadget
        The fault-tolerant gadget to adapt.
    code : Code
        The code this gadget operates on.
    block_name : str
        Name for the code block (used in layout).
        
    Attributes
    ----------
    ft_gadget : FTGadget
        Reference to wrapped gadget.
    code : Code
        Reference to code.
    """
    
    def __init__(
        self,
        ft_gadget: "FTGadget",
        code: "Code",
        block_name: str = "block_0",
    ):
        self.ft_gadget = ft_gadget
        self.code = code
        self.block_name = block_name
        
        # Create layout for the gadget
        self._layout = None
        self._scheduler = None
    
    @property
    def name(self) -> str:
        """Name based on wrapped gadget."""
        ft_name = getattr(self.ft_gadget, 'gate_name', None)
        if ft_name:
            return f"Adapted_{ft_name}"
        return f"Adapted_{type(self.ft_gadget).__name__}"
    
    @property
    def requires_ancillas(self) -> bool:
        """Check if wrapped gadget needs ancillas."""
        return getattr(self.ft_gadget, 'requires_ancillas', False)
    
    @property
    def ancillas_per_block(self) -> int:
        """Number of ancillas needed."""
        return getattr(self.ft_gadget, 'ancillas_per_block', 0)
    
    def _ensure_layout(self, data_qubits: List[int]) -> None:
        """Create layout and scheduler if not already done."""
        if self._layout is not None:
            return
        
        try:
            from qectostim.gadgets.layout import GadgetLayout
            from qectostim.gadgets.scheduling import GadgetScheduler
            
            self._layout = GadgetLayout()
            self._layout.add_block(
                name=self.block_name,
                code=self.code,
                data_qubits=data_qubits,
            )
            self._scheduler = GadgetScheduler(self._layout)
        except ImportError:
            # Layout/scheduler not available, use simpler emission
            self._layout = "simple"
    
    def emit(
        self,
        circuit: stim.Circuit,
        level: int = 0,
        block_idx: int = 0,
        data_qubits: Optional[List[int]] = None,
        noise_model: Optional["NoiseModel"] = None,
        measurement_offset: int = 0,
        **kwargs,
    ) -> MeasurementMap:
        """
        Emit the FT gadget operations using EC interface.
        
        Translates between FT gadget's phase-based emission and
        EC gadget's single-shot emission.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to append to.
        level : int
            Hierarchical level.
        block_idx : int
            Block index within level.
        data_qubits : List[int], optional
            Data qubit indices.
        noise_model : NoiseModel, optional
            Noise model to apply.
        measurement_offset : int
            Starting measurement index.
            
        Returns
        -------
        MeasurementMap
            Measurement tracking (may be empty for pure gate gadgets).
        """
        n = self.code.n
        if data_qubits is None:
            data_qubits = list(range(n))
        
        mmap = MeasurementMap(offset=measurement_offset)
        meas_count = 0
        
        # Check if FT gadget has phase-based interface
        if hasattr(self.ft_gadget, 'emit_next_phase'):
            # Phase-based emission
            meas_count = self._emit_phased(circuit, data_qubits, measurement_offset)
        elif hasattr(self.ft_gadget, 'emit'):
            # Direct emission (simpler gadgets)
            meas_count = self._emit_direct(circuit, data_qubits, measurement_offset)
        else:
            # Fallback: assume it's a transversal gate
            meas_count = self._emit_transversal(circuit, data_qubits)
        
        mmap.total_measurements = meas_count
        return mmap
    
    def _emit_phased(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        measurement_offset: int,
    ) -> int:
        """Emit using phase-based interface."""
        from qectostim.gadgets.base import PhaseType
        
        total_meas = 0
        phase_idx = 0
        max_phases = 10  # Safety limit
        
        while phase_idx < max_phases:
            # Get targets for this phase
            targets = self._make_physical_targets(data_qubits)
            
            # Emit phase
            result = self.ft_gadget.emit_next_phase(circuit, targets, phase_idx)
            
            total_meas += getattr(result, 'measurement_count', 0)
            
            if result.is_final or result.phase_type == PhaseType.COMPLETE:
                break
            
            # Add tick between phases
            circuit.append("TICK")
            phase_idx += 1
        
        return total_meas
    
    def _emit_direct(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
        measurement_offset: int,
    ) -> int:
        """Emit using direct emit() method."""
        targets = self._make_physical_targets(data_qubits)
        
        # Some gadgets have emit(circuit, targets)
        try:
            result = self.ft_gadget.emit(circuit, targets)
            return getattr(result, 'measurement_count', 0)
        except TypeError:
            # Try without targets
            result = self.ft_gadget.emit(circuit)
            return getattr(result, 'measurement_count', 0)
    
    def _emit_transversal(
        self,
        circuit: stim.Circuit,
        data_qubits: List[int],
    ) -> int:
        """Emit as transversal gate."""
        gate_name = getattr(self.ft_gadget, 'gate_name', 'H')
        
        # Map Stim gate names
        stim_gates = {
            'H': 'H', 'S': 'S', 'S_DAG': 'S_DAG',
            'X': 'X', 'Y': 'Y', 'Z': 'Z',
            'CNOT': 'CX', 'CX': 'CX', 'CZ': 'CZ',
        }
        
        stim_gate = stim_gates.get(gate_name.upper(), gate_name)
        
        if stim_gate in ('CX', 'CZ', 'CNOT'):
            # Two-qubit gate - needs paired blocks
            # For now, just skip
            pass
        else:
            # Single-qubit transversal gate
            circuit.append(stim_gate, data_qubits)
        
        return 0
    
    def _make_physical_targets(self, data_qubits: List[int]) -> Any:
        """Create PhysicalTargets for FT gadget."""
        try:
            from qectostim.gadgets.layout import QubitIndexMap
            
            return QubitIndexMap(
                data=data_qubits,
                x_ancilla=[],
                z_ancilla=[],
            )
        except ImportError:
            # Return simple dict
            return {'data': data_qubits}
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        """Get syndrome schedule (usually empty for gate gadgets)."""
        return SyndromeSchedule(
            stabilizer_types=[],
            rounds_per_type={},
            schedule=[],
        )
    
    def get_logical_map(
        self,
        inner_code: Any,
        outer_code: Any,
    ) -> LogicalMeasurementMap:
        """Get logical measurement mapping."""
        return LogicalMeasurementMap()
    
    def get_stabilizer_transform(self) -> Optional[Any]:
        """
        Get stabilizer transform from wrapped gadget.
        
        This allows experiments to track how stabilizers change
        across the gadget for proper detector construction.
        """
        if hasattr(self.ft_gadget, 'stabilizer_transform'):
            return self.ft_gadget.stabilizer_transform()
        return None


def adapt_ft_gadget(
    ft_gadget: "FTGadget",
    code: "Code",
    block_name: str = "block_0",
) -> FTGadgetAdapter:
    """
    Factory function to create an FT gadget adapter.
    
    Parameters
    ----------
    ft_gadget : FTGadget
        The fault-tolerant gadget from gadgets/.
    code : Code
        The code the gadget operates on.
    block_name : str
        Name for the code block.
        
    Returns
    -------
    FTGadgetAdapter
        Adapted gadget usable in EC experiments.
        
    Example
    -------
    >>> from qectostim.gadgets.transversal import TransversalHadamard
    >>> from qectostim.experiments.gadgets.ft_gadget_adapter import adapt_ft_gadget
    >>> 
    >>> h_gadget = adapt_ft_gadget(TransversalHadamard(), steane_code)
    >>> mmap = h_gadget.emit(circuit, level=0, block_idx=0)
    """
    return FTGadgetAdapter(ft_gadget, code, block_name)


class ScheduledECGadget(ECGadget):
    """
    EC Gadget that uses GadgetScheduler for parallel circuit emission.
    
    This is a bridge class that allows EC gadgets to leverage the
    sophisticated scheduling from gadgets/scheduling.py.
    
    Usage:
        Override `schedule_operations()` to add operations to the scheduler,
        then call `emit()` to generate the parallel circuit.
    """
    
    def __init__(self, code: Any):
        self.code = code
        self._scheduler = None
    
    @property
    def name(self) -> str:
        return "ScheduledEC"
    
    @property
    def requires_ancillas(self) -> bool:
        return True
    
    @property
    def ancillas_per_block(self) -> int:
        return 0
    
    def _create_scheduler(self, data_qubits: List[int]) -> Any:
        """Create a GadgetScheduler for this emission."""
        try:
            from qectostim.gadgets.layout import GadgetLayout
            from qectostim.gadgets.scheduling import GadgetScheduler
            
            layout = GadgetLayout()
            layout.add_block(
                name="block",
                code=self.code,
                data_qubits=data_qubits,
            )
            return GadgetScheduler(layout)
        except ImportError:
            return None
    
    def schedule_operations(
        self,
        scheduler: Any,
        data_qubits: List[int],
        ancilla_qubits: List[int],
    ) -> int:
        """
        Override this to schedule operations.
        
        Returns number of measurements.
        """
        raise NotImplementedError("Subclass must implement schedule_operations")
    
    def emit(
        self,
        circuit: stim.Circuit,
        level: int = 0,
        block_idx: int = 0,
        data_qubits: Optional[List[int]] = None,
        ancilla_qubits: Optional[List[int]] = None,
        measurement_offset: int = 0,
        **kwargs,
    ) -> MeasurementMap:
        """Emit using scheduler."""
        if data_qubits is None:
            data_qubits = list(range(self.code.n))
        if ancilla_qubits is None:
            ancilla_qubits = []
        
        scheduler = self._create_scheduler(data_qubits)
        
        if scheduler is None:
            # Fallback to simple emission
            return self._emit_simple(circuit, level, block_idx, data_qubits, ancilla_qubits, measurement_offset)
        
        meas_count = self.schedule_operations(scheduler, data_qubits, ancilla_qubits)
        
        # Emit scheduled circuit
        sched_circuit = scheduler.to_stim()
        circuit += sched_circuit
        
        mmap = MeasurementMap(offset=measurement_offset)
        mmap.total_measurements = meas_count
        return mmap
    
    def _emit_simple(
        self,
        circuit: stim.Circuit,
        level: int,
        block_idx: int,
        data_qubits: List[int],
        ancilla_qubits: List[int],
        measurement_offset: int,
    ) -> MeasurementMap:
        """Simple fallback emission."""
        return MeasurementMap(offset=measurement_offset)
    
    def get_syndrome_schedule(self) -> SyndromeSchedule:
        return SyndromeSchedule(stabilizer_types=[], rounds_per_type={}, schedule=[])
    
    def get_logical_map(self, inner_code: Any, outer_code: Any) -> LogicalMeasurementMap:
        return LogicalMeasurementMap()
