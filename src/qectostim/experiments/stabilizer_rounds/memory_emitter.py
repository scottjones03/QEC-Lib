# src/qectostim/experiments/stabilizer_rounds/memory_emitter.py
"""
Memory Round Emitter for FT Gadget Experiments.

This module extracts the memory round emission logic from FaultTolerantGadgetExperiment
into a dedicated, reusable component. The emitter handles:

1. Pre-gadget memory rounds (with anchor detectors based on initial state)
2. Post-gadget memory rounds (with crossing detectors based on gate transform)
3. Parallel vs sequential extraction strategies

The key insight is that memory round emission depends on:
- PreparationConfig: Which stabilizers are deterministic (for anchors)
- CrossingDetectorConfig: How to emit crossing detectors after the gate
- Gadget.requires_parallel_extraction(): Whether blocks are measured together

By centralizing this logic, we enable:
- Clean separation of concerns from the experiment class
- Reuse across different gadget types
- Consistent behavior matching ground truth builders
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, TYPE_CHECKING

import stim

if TYPE_CHECKING:
    from qectostim.experiments.stabilizer_rounds import DetectorContext
    from qectostim.experiments.stabilizer_rounds.base import BaseStabilizerRoundBuilder
    from qectostim.gadgets.base import (
        PreparationConfig,
        CrossingDetectorConfig,
        CrossingDetectorFormula,
        CrossingDetectorTerm,
    )


@dataclass
class RoundEmissionConfig:
    """
    Configuration for memory round emission.
    
    Attributes
    ----------
    num_rounds : int
        Number of stabilizer rounds to emit.
    use_parallel_extraction : bool
        If True, emit both blocks together per round (Z_all, X_all).
        If False, emit each block sequentially (block_0 all, block_1 all).
    blocks_to_skip : Set[str]
        Block names to skip during emission.
    prep_config : Optional[PreparationConfig]
        Configuration for anchor detector determination.
    crossing_config : Optional[CrossingDetectorConfig]
        Configuration for crossing detectors (post-gadget only).
    pre_gadget_meas : Optional[Dict[str, Dict[str, List[int]]]]
        Pre-gadget measurement indices for crossing detector construction.
    emit_crossing_on_first : bool
        If True, emit crossing detectors on first round instead of temporal.
    """
    num_rounds: int
    use_parallel_extraction: bool = False
    blocks_to_skip: Set[str] = None
    prep_config: Optional["PreparationConfig"] = None
    crossing_config: Optional["CrossingDetectorConfig"] = None
    pre_gadget_meas: Optional[Dict[str, Dict[str, List[int]]]] = None
    emit_crossing_on_first: bool = False
    
    def __post_init__(self):
        if self.blocks_to_skip is None:
            self.blocks_to_skip = set()


class MemoryRoundEmitter:
    """
    Emitter for stabilizer memory rounds in FT gadget experiments.
    
    Handles pre-gadget and post-gadget memory round emission with:
    - Anchor detectors based on initial state determinism
    - Crossing detectors based on gate stabilizer transform
    - Parallel or sequential block extraction
    
    This class extracts and centralizes the logic from:
    - FaultTolerantGadgetExperiment._emit_pre_gadget_memory()
    - FaultTolerantGadgetExperiment._emit_post_gadget_memory()
    - FaultTolerantGadgetExperiment._emit_parallel_pre_gadget_rounds()
    - FaultTolerantGadgetExperiment._emit_parallel_post_gadget_rounds()
    """
    
    def __init__(
        self,
        builders: List["BaseStabilizerRoundBuilder"],
        ctx: "DetectorContext",
    ):
        """
        Initialize the emitter.
        
        Parameters
        ----------
        builders : List[BaseStabilizerRoundBuilder]
            Round builders for each code block.
        ctx : DetectorContext
            Detector context for measurement tracking.
        """
        self.builders = builders
        self.ctx = ctx
        self._block_builders = {b.block_name: b for b in builders}
    
    def emit_pre_gadget_rounds(
        self,
        circuit: stim.Circuit,
        config: RoundEmissionConfig,
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Emit pre-gadget stabilizer rounds.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        config : RoundEmissionConfig
            Configuration for round emission.
            
        Returns
        -------
        Dict[str, Dict[str, List[int]]]
            Pre-gadget measurement indices per block:
            {block_name: {'x': [...], 'z': [...], 'round': int}}
        """
        if config.use_parallel_extraction:
            return self._emit_parallel_rounds(circuit, config, is_pre_gadget=True)
        else:
            return self._emit_sequential_rounds(circuit, config, is_pre_gadget=True)
    
    def emit_post_gadget_rounds(
        self,
        circuit: stim.Circuit,
        config: RoundEmissionConfig,
        destroyed_blocks: Optional[Set[str]] = None,
    ) -> None:
        """
        Emit post-gadget stabilizer rounds with crossing detectors.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        config : RoundEmissionConfig
            Configuration for round emission (includes crossing config).
        destroyed_blocks : Optional[Set[str]]
            Blocks that are destroyed during the gadget (skip these).
        """
        if destroyed_blocks is not None:
            config.blocks_to_skip = config.blocks_to_skip | destroyed_blocks
        
        if config.use_parallel_extraction:
            self._emit_parallel_rounds(circuit, config, is_pre_gadget=False)
        else:
            self._emit_sequential_rounds(circuit, config, is_pre_gadget=False)
    
    def _emit_sequential_rounds(
        self,
        circuit: stim.Circuit,
        config: RoundEmissionConfig,
        is_pre_gadget: bool,
    ) -> Dict[str, Dict[str, List[int]]]:
        """Emit rounds with sequential per-block extraction."""
        from qectostim.experiments.preparation import get_deterministic_stabilizers
        from qectostim.experiments.stabilizer_rounds import StabilizerBasis
        
        last_meas = {}
        
        for round_idx in range(config.num_rounds):
            for builder in self.builders:
                if builder.block_name in config.blocks_to_skip:
                    continue
                
                # Determine anchor flags for first round (pre-gadget only)
                emit_z_anchors = False
                emit_x_anchors = False
                explicit_anchor_mode = False
                
                if is_pre_gadget and round_idx == 0 and config.prep_config is not None:
                    determinism = get_deterministic_stabilizers(
                        config.prep_config, builder.block_name
                    )
                    emit_z_anchors = determinism["Z"]
                    emit_x_anchors = determinism["X"]
                    explicit_anchor_mode = True
                
                # Emit crossing detectors on first post-gadget round?
                emit_detectors = True
                if not is_pre_gadget and round_idx == 0 and config.emit_crossing_on_first:
                    emit_detectors = False  # We'll emit crossing detectors manually
                
                builder.emit_round(
                    circuit,
                    StabilizerBasis.BOTH,
                    emit_detectors=emit_detectors,
                    emit_z_anchors=emit_z_anchors,
                    emit_x_anchors=emit_x_anchors,
                    explicit_anchor_mode=explicit_anchor_mode,
                )
            
            # Emit crossing detectors after first post-gadget round
            if (not is_pre_gadget and round_idx == 0 and 
                config.emit_crossing_on_first and config.crossing_config is not None):
                self._emit_crossing_detectors(
                    circuit, config.crossing_config, config.pre_gadget_meas,
                    config.blocks_to_skip
                )
        
        # Record final measurement indices
        for builder in self.builders:
            if builder.block_name not in config.blocks_to_skip:
                if hasattr(builder, 'get_last_measurement_indices'):
                    last_meas[builder.block_name] = builder.get_last_measurement_indices()
        
        return last_meas
    
    def _emit_parallel_rounds(
        self,
        circuit: stim.Circuit,
        config: RoundEmissionConfig,
        is_pre_gadget: bool,
    ) -> Dict[str, Dict[str, List[int]]]:
        """
        Emit rounds with parallel extraction (both blocks per round).
        
        Measurement order per round: [D_Z, A_Z, D_X, A_X]
        This matches ground truth builder ordering.
        """
        from qectostim.experiments.preparation import get_deterministic_stabilizers
        from qectostim.experiments.stabilizer_rounds import StabilizerBasis
        
        active_builders = [
            b for b in self.builders 
            if b.block_name not in config.blocks_to_skip
        ]
        last_meas = {}
        
        for round_idx in range(config.num_rounds):
            # Compute anchor info for first round
            anchor_info = {}
            for builder in active_builders:
                emit_z_anchors = False
                emit_x_anchors = False
                if is_pre_gadget and round_idx == 0 and config.prep_config is not None:
                    determinism = get_deterministic_stabilizers(
                        config.prep_config, builder.block_name
                    )
                    emit_z_anchors = determinism["Z"]
                    emit_x_anchors = determinism["X"]
                anchor_info[builder.block_name] = {'z': emit_z_anchors, 'x': emit_x_anchors}
            
            # Determine if we emit detectors automatically
            emit_auto_detectors = True
            if not is_pre_gadget and round_idx == 0 and config.emit_crossing_on_first:
                emit_auto_detectors = False
            
            # For the first pre-gadget round with anchors, we need special ordering:
            # - Blocks with X anchors (|+⟩ prep): X must be measured BEFORE Z
            # - Blocks with Z anchors (|0⟩ prep): Z must be measured BEFORE X
            # This ensures anchors are measured before the other basis disturbs the state.
            #
            # For subsequent rounds, order doesn't matter (both bases have history).
            first_prep_round = is_pre_gadget and round_idx == 0 and config.prep_config is not None
            
            if first_prep_round:
                # Separate blocks by which basis needs to be measured first
                x_first_blocks = [b for b in active_builders 
                                  if anchor_info.get(b.block_name, {}).get('x', False)]
                z_first_blocks = [b for b in active_builders
                                  if anchor_info.get(b.block_name, {}).get('z', False)]
                # Blocks with neither anchor can go in any order
                no_anchor_blocks = [b for b in active_builders
                                    if b not in x_first_blocks and b not in z_first_blocks]
                
                # Emit X layer for X-first blocks
                for builder in x_first_blocks:
                    if hasattr(builder, 'emit_x_layer'):
                        builder.emit_x_layer(
                            circuit, emit_detectors=emit_auto_detectors, emit_x_anchors=True)
                    else:
                        builder.emit_round(
                            circuit, StabilizerBasis.X, emit_detectors=emit_auto_detectors,
                            emit_x_anchors=True, explicit_anchor_mode=True)
                
                # Emit Z layer for Z-first blocks
                for builder in z_first_blocks:
                    if hasattr(builder, 'emit_z_layer'):
                        builder.emit_z_layer(
                            circuit, emit_detectors=emit_auto_detectors, emit_z_anchors=True)
                    else:
                        builder.emit_round(
                            circuit, StabilizerBasis.Z, emit_detectors=emit_auto_detectors,
                            emit_z_anchors=True, explicit_anchor_mode=True)
                
                # Emit Z layer for X-first blocks (after their X layer)
                for builder in x_first_blocks:
                    if hasattr(builder, 'emit_z_layer'):
                        builder.emit_z_layer(
                            circuit, emit_detectors=emit_auto_detectors, emit_z_anchors=False)
                    else:
                        builder.emit_round(
                            circuit, StabilizerBasis.Z, emit_detectors=emit_auto_detectors,
                            emit_z_anchors=False, explicit_anchor_mode=True)
                
                # Emit X layer for Z-first blocks (after their Z layer)
                for builder in z_first_blocks:
                    if hasattr(builder, 'emit_x_layer'):
                        builder.emit_x_layer(
                            circuit, emit_detectors=emit_auto_detectors, emit_x_anchors=False)
                    else:
                        builder.emit_round(
                            circuit, StabilizerBasis.X, emit_detectors=emit_auto_detectors,
                            emit_x_anchors=False, explicit_anchor_mode=True)
                
                # No-anchor blocks: standard order (Z then X)
                for builder in no_anchor_blocks:
                    if hasattr(builder, 'emit_z_layer'):
                        builder.emit_z_layer(circuit, emit_detectors=emit_auto_detectors, emit_z_anchors=False)
                    else:
                        builder.emit_round(circuit, StabilizerBasis.Z, emit_detectors=emit_auto_detectors,
                                         emit_z_anchors=False, explicit_anchor_mode=True)
                for builder in no_anchor_blocks:
                    if hasattr(builder, 'emit_x_layer'):
                        builder.emit_x_layer(circuit, emit_detectors=emit_auto_detectors, emit_x_anchors=False)
                    else:
                        builder.emit_round(circuit, StabilizerBasis.X, emit_detectors=emit_auto_detectors,
                                         emit_x_anchors=False, explicit_anchor_mode=True)
            else:
                # Standard ordering for non-first rounds: Z then X
                # === Z LAYER: All blocks together ===
                for builder in active_builders:
                    z_anchors = anchor_info.get(builder.block_name, {}).get('z', False)
                    if hasattr(builder, 'emit_z_layer'):
                        builder.emit_z_layer(
                            circuit, 
                            emit_detectors=emit_auto_detectors,
                            emit_z_anchors=z_anchors,
                        )
                    else:
                        builder.emit_round(
                            circuit, StabilizerBasis.Z, 
                            emit_detectors=emit_auto_detectors,
                            emit_z_anchors=z_anchors,
                            explicit_anchor_mode=(is_pre_gadget and round_idx == 0),
                        )
                
                # === X LAYER: All blocks together ===
                for builder in active_builders:
                    x_anchors = anchor_info.get(builder.block_name, {}).get('x', False)
                    if hasattr(builder, 'emit_x_layer'):
                        builder.emit_x_layer(
                            circuit,
                            emit_detectors=emit_auto_detectors,
                            emit_x_anchors=x_anchors,
                        )
                    else:
                        builder.emit_round(
                            circuit, StabilizerBasis.X,
                            emit_detectors=emit_auto_detectors,
                            emit_x_anchors=x_anchors,
                            explicit_anchor_mode=(is_pre_gadget and round_idx == 0),
                        )
            
            # === Finalize round ===
            for builder in active_builders:
                if hasattr(builder, 'finalize_parallel_round'):
                    builder.finalize_parallel_round(circuit)
            
            # Emit crossing detectors after first post-gadget round
            if (not is_pre_gadget and round_idx == 0 and 
                config.emit_crossing_on_first and config.crossing_config is not None):
                self._emit_crossing_detectors(
                    circuit, config.crossing_config, config.pre_gadget_meas,
                    config.blocks_to_skip
                )
        
        # Record final measurement indices
        for builder in active_builders:
            if hasattr(builder, 'get_last_measurement_indices'):
                last_meas[builder.block_name] = builder.get_last_measurement_indices()
        
        return last_meas
    
    def _emit_crossing_detectors(
        self,
        circuit: stim.Circuit,
        crossing_config: "CrossingDetectorConfig",
        pre_gadget_meas: Optional[Dict[str, Dict[str, List[int]]]],
        destroyed_blocks: Set[str],
    ) -> None:
        """
        Emit crossing detectors using pre-gate and post-gate measurements.
        
        Parameters
        ----------
        circuit : stim.Circuit
            Circuit to emit into.
        crossing_config : CrossingDetectorConfig
            Configuration specifying crossing detector formulas.
        pre_gadget_meas : Dict[str, Dict[str, List[int]]]
            Pre-gadget measurement indices per block.
        destroyed_blocks : Set[str]
            Blocks that are destroyed (skip these in post-meas lookup).
        """
        if pre_gadget_meas is None:
            return
        
        # Get post-gadget measurement indices
        post_gadget_meas = {}
        for builder in self.builders:
            if builder.block_name not in destroyed_blocks:
                if hasattr(builder, 'get_last_measurement_indices'):
                    post_gadget_meas[builder.block_name] = builder.get_last_measurement_indices()
        
        total_meas = circuit.num_measurements
        
        # Emit each crossing detector formula
        for formula in crossing_config.formulas:
            if not formula.terms:
                continue
            
            # Get first term to determine stabilizer count and coordinates
            first_term = formula.terms[0]
            first_block = first_term.block
            first_basis = first_term.stabilizer_type.upper()
            
            # Get stabilizer count from builder
            if first_block not in self._block_builders:
                continue
            builder = self._block_builders[first_block]
            
            if first_basis == 'X':
                num_stab = len(builder.x_ancillas) if hasattr(builder, 'x_ancillas') else 0
            else:
                num_stab = len(builder.z_ancillas) if hasattr(builder, 'z_ancillas') else 0
            
            # Emit detector for each stabilizer
            for stab_idx in range(num_stab):
                targets = []
                
                for term in formula.terms:
                    block = term.block
                    basis = term.stabilizer_type.lower()
                    phase = term.timing
                    
                    # Get measurement indices
                    if phase == 'pre':
                        meas_dict = pre_gadget_meas.get(block, {})
                    else:
                        meas_dict = post_gadget_meas.get(block, {})
                    
                    meas_list = meas_dict.get(basis, [])
                    
                    if stab_idx < len(meas_list):
                        abs_idx = meas_list[stab_idx]
                        if abs_idx is not None:
                            lookback = abs_idx - total_meas
                            targets.append(stim.target_rec(lookback))
                
                if targets:
                    coords = self._get_detector_coords(builder, first_basis, stab_idx)
                    circuit.append("DETECTOR", targets, coords)
    
    def _get_detector_coords(
        self,
        builder: "BaseStabilizerRoundBuilder",
        basis: str,
        stab_idx: int,
    ) -> List[float]:
        """Get detector coordinates for a stabilizer."""
        if basis.upper() == 'X':
            ancillas = builder.x_ancillas if hasattr(builder, 'x_ancillas') else []
        else:
            ancillas = builder.z_ancillas if hasattr(builder, 'z_ancillas') else []
        
        if stab_idx < len(ancillas):
            qubit_idx = ancillas[stab_idx]
            if hasattr(builder, 'qubit_coords') and qubit_idx in builder.qubit_coords:
                coords = list(builder.qubit_coords[qubit_idx])
                round_num = getattr(builder, '_round_number', 0)
                coords.append(float(round_num))
                return coords
        
        return []
    
    def get_pre_gadget_measurement_indices(self) -> Dict[str, Dict[str, List[int]]]:
        """
        Get the current measurement indices for all builders.
        
        Call this after emitting pre-gadget rounds to save the indices
        for crossing detector construction.
        
        Returns
        -------
        Dict[str, Dict[str, List[int]]]
            Measurement indices: {block_name: {'x': [...], 'z': [...]}}
        """
        result = {}
        for builder in self.builders:
            if hasattr(builder, 'get_last_measurement_indices'):
                result[builder.block_name] = builder.get_last_measurement_indices()
        return result


# Convenience factory function
def create_memory_emitter(
    builders: List["BaseStabilizerRoundBuilder"],
    ctx: "DetectorContext",
) -> MemoryRoundEmitter:
    """
    Create a MemoryRoundEmitter for the given builders.
    
    Parameters
    ----------
    builders : List[BaseStabilizerRoundBuilder]
        Round builders for each code block.
    ctx : DetectorContext
        Detector context for measurement tracking.
        
    Returns
    -------
    MemoryRoundEmitter
        Configured emitter.
    """
    return MemoryRoundEmitter(builders, ctx)
