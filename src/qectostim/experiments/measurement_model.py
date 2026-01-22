# src/qectostim/experiments/measurement_model.py
"""
Measurement-Centric Data Model for Hierarchical Concatenated Codes.

This module defines the core data structures for tracking measurements in a
way that directly maps to decoder needs. The key insight is that every inner
EC application creates measurement data that the decoder needs, regardless of
circuit structure.

Key Concepts
============

1. **InnerECInstance**: One per (block_id, EC-application)
   - Every time inner EC is applied to a block, we create one instance
   - Contains the X and Z ancilla measurement indices for that EC
   - Decoder iterates over these to compute inner corrections

2. **OuterSyndromeValue**: One per outer stabilizer measurement
   - Links to the ancilla_ec_instance_id for that gadget
   - Contains the transversal measurement indices
   - Decoder uses ancilla's inner correction directly

3. **FinalDataMeasurement**: One per data block at the end
   - Contains the transversal measurement indices
   - Decoder knows all EC instances for each data block via block_to_ec_instances

4. **block_to_ec_instances**: Dict[int, List[int]]
   - Maps each block_id to ALL InnerECInstance IDs that touched it
   - Decoder uses this to accumulate corrections for Phase 3

Rationale
=========

OLD MODEL (segment-based):
- Segments conflate circuit structure with decoder needs
- outer_stab segments contain ancilla EC, but decoder looked in inner_ec segments
- Required complex mapping between segment types

NEW MODEL (measurement-centric):
- Every EC application creates an InnerECInstance regardless of context
- OuterSyndromeValue directly references its ancilla's EC instance
- No ambiguity about where ancilla corrections come from
- Simpler, more direct mapping to decoder phases

Example Flow
============

For one outer round with an X stabilizer:

1. Ancilla prepared in |+⟩
2. Inner EC applied to ancilla → InnerECInstance(id=42, block_id=7, ...)
3. CNOT gates executed
4. Inner EC applied to ancilla → InnerECInstance(id=43, block_id=7, ...)
5. Transversal measurement → OuterSyndromeValue(..., ancilla_ec_instance_id=43)

The decoder:
- Phase 1: Processes InnerECInstance 42 and 43, computes corrections
- Phase 2: For OuterSyndromeValue, gets correction from instance 43 directly

This is much cleaner than trying to find "which inner_ec segment has this ancilla".

"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class InnerECInstance:
    """
    One application of inner EC to a specific block (with multiple syndrome rounds).
    
    This is the fundamental unit of inner syndrome measurement. Each instance
    represents d rounds of syndrome extraction on one block, where d is the
    inner code distance. This aggregation enables proper temporal DEM decoding.
    
    The decoder iterates over these to compute inner corrections:
    - X ancilla measurements detect Z errors → z_correction
    - Z ancilla measurements detect X errors → x_correction
    
    Attributes
    ----------
    instance_id : int
        Global unique ID for this EC instance (sequential counter)
    block_id : int  
        Which block this EC was applied to
    x_anc_measurements : List[int]
        Measurement indices for X ancilla qubits across all n_rounds
        (length = n_x_stabs * n_rounds)
    z_anc_measurements : List[int]
        Measurement indices for Z ancilla qubits across all n_rounds
        (length = n_z_stabs * n_rounds)
    context : str
        Where this EC occurred: 'init', 'pre_outer', 'intra_gadget', 'post_outer', 'final'
    outer_round : int
        Which outer round this belongs to (-1 for init/final)
    n_rounds : int
        Number of syndrome extraction rounds in this EC instance.
        Must be >= 2 for temporal DEM, >= 3 for majority vote.
        Default is 1 for backward compatibility, but this should be
        set to inner code distance (d) for proper error correction.
    """
    instance_id: int
    block_id: int
    x_anc_measurements: List[int] = field(default_factory=list)
    z_anc_measurements: List[int] = field(default_factory=list)
    context: str = "unknown"
    outer_round: int = -1
    n_rounds: int = 1


@dataclass
class OuterSyndromeValue:
    """
    One outer stabilizer syndrome measurement.
    
    Represents a single logical measurement of an ancilla block that
    encodes one bit of outer syndrome information.
    
    Attributes
    ----------
    outer_round : int
        Which outer round this measurement belongs to
    stab_type : str
        'X' or 'Z' - type of outer stabilizer being measured
    stab_idx : int
        Index of the stabilizer within its type (0, 1, 2, ...)
    ancilla_block_id : int
        Which ancilla block encodes this measurement
    transversal_meas_indices : List[int]
        The n physical measurement indices for the ancilla data qubits
    ancilla_ec_instance_id : int
        ID of the InnerECInstance that protects this measurement
        (the EC applied to the ancilla just before measurement)
    """
    outer_round: int
    stab_type: str  # 'X' or 'Z'
    stab_idx: int
    ancilla_block_id: int
    transversal_meas_indices: List[int] = field(default_factory=list)
    ancilla_ec_instance_id: int = -1  # -1 means no EC protection


@dataclass
class FinalDataMeasurement:
    """
    Final transversal measurement of a data block.
    
    Attributes
    ----------
    block_id : int
        Which data block this is
    transversal_meas_indices : List[int]
        The n physical measurement indices
    measurement_basis : str
        'X' or 'Z' - which basis was used
    """
    block_id: int
    transversal_meas_indices: List[int] = field(default_factory=list)
    measurement_basis: str = "Z"


@dataclass
class MeasurementCentricMetadata:
    """
    Complete decoder metadata in measurement-centric format.
    
    This replaces the old segment-based metadata format. All mappings
    are designed for direct use by the decoder's 4-phase pipeline.
    
    Attributes
    ----------
    inner_ec_instances : Dict[int, InnerECInstance]
        All EC instances keyed by instance_id
    
    block_to_ec_instances : Dict[int, List[int]]
        Maps block_id -> list of instance_ids that touched that block
        Used by Phase 3 to accumulate all corrections for each data block
    
    outer_syndrome_values : List[OuterSyndromeValue]
        All outer syndrome measurements in order
        Each has ancilla_ec_instance_id for Phase 2 correction lookup
    
    final_data_measurements : Dict[int, FinalDataMeasurement]
        Final measurements keyed by block_id
        Used by Phase 3 and Phase 4
    
    n_x_stabs_inner : int
        Number of X stabilizers in inner code
    
    n_z_stabs_inner : int
        Number of Z stabilizers in inner code
    
    n_x_stabs_outer : int
        Number of X stabilizers in outer code
    
    n_z_stabs_outer : int
        Number of Z stabilizers in outer code
    """
    inner_ec_instances: Dict[int, InnerECInstance] = field(default_factory=dict)
    block_to_ec_instances: Dict[int, List[int]] = field(default_factory=dict)
    outer_syndrome_values: List[OuterSyndromeValue] = field(default_factory=list)
    final_data_measurements: Dict[int, FinalDataMeasurement] = field(default_factory=dict)
    n_x_stabs_inner: int = 0
    n_z_stabs_inner: int = 0
    n_x_stabs_outer: int = 0
    n_z_stabs_outer: int = 0
    
    def add_ec_instance(self, instance: InnerECInstance) -> None:
        """Add an EC instance and update mappings."""
        self.inner_ec_instances[instance.instance_id] = instance
        if instance.block_id not in self.block_to_ec_instances:
            self.block_to_ec_instances[instance.block_id] = []
        self.block_to_ec_instances[instance.block_id].append(instance.instance_id)
    
    def add_outer_syndrome(self, syndrome: OuterSyndromeValue) -> None:
        """Add an outer syndrome measurement."""
        self.outer_syndrome_values.append(syndrome)
    
    def add_final_measurement(self, measurement: FinalDataMeasurement) -> None:
        """Add a final data measurement."""
        self.final_data_measurements[measurement.block_id] = measurement
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'inner_ec_instances': {
                k: {
                    'instance_id': v.instance_id,
                    'block_id': v.block_id,
                    'x_anc_measurements': v.x_anc_measurements,
                    'z_anc_measurements': v.z_anc_measurements,
                    'context': v.context,
                    'outer_round': v.outer_round,
                    'n_rounds': v.n_rounds,
                }
                for k, v in self.inner_ec_instances.items()
            },
            'block_to_ec_instances': dict(self.block_to_ec_instances),
            'outer_syndrome_values': [
                {
                    'outer_round': v.outer_round,
                    'stab_type': v.stab_type,
                    'stab_idx': v.stab_idx,
                    'ancilla_block_id': v.ancilla_block_id,
                    'transversal_meas_indices': v.transversal_meas_indices,
                    'ancilla_ec_instance_id': v.ancilla_ec_instance_id,
                }
                for v in self.outer_syndrome_values
            ],
            'final_data_measurements': {
                k: {
                    'block_id': v.block_id,
                    'transversal_meas_indices': v.transversal_meas_indices,
                    'measurement_basis': v.measurement_basis,
                }
                for k, v in self.final_data_measurements.items()
            },
            'n_x_stabs_inner': self.n_x_stabs_inner,
            'n_z_stabs_inner': self.n_z_stabs_inner,
            'n_x_stabs_outer': self.n_x_stabs_outer,
            'n_z_stabs_outer': self.n_z_stabs_outer,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MeasurementCentricMetadata":
        """Reconstruct from dictionary."""
        metadata = cls()
        metadata.n_x_stabs_inner = data.get('n_x_stabs_inner', 0)
        metadata.n_z_stabs_inner = data.get('n_z_stabs_inner', 0)
        metadata.n_x_stabs_outer = data.get('n_x_stabs_outer', 0)
        metadata.n_z_stabs_outer = data.get('n_z_stabs_outer', 0)
        
        # Reconstruct EC instances
        for k, v in data.get('inner_ec_instances', {}).items():
            instance = InnerECInstance(
                instance_id=v['instance_id'],
                block_id=v['block_id'],
                x_anc_measurements=v.get('x_anc_measurements', []),
                z_anc_measurements=v.get('z_anc_measurements', []),
                context=v.get('context', 'unknown'),
                outer_round=v.get('outer_round', -1),
                n_rounds=v.get('n_rounds', 1),
            )
            metadata.inner_ec_instances[int(k)] = instance
        
        # Reconstruct block mapping
        metadata.block_to_ec_instances = {
            int(k): v for k, v in data.get('block_to_ec_instances', {}).items()
        }
        
        # Reconstruct outer syndromes
        for v in data.get('outer_syndrome_values', []):
            syndrome = OuterSyndromeValue(
                outer_round=v['outer_round'],
                stab_type=v['stab_type'],
                stab_idx=v['stab_idx'],
                ancilla_block_id=v['ancilla_block_id'],
                transversal_meas_indices=v.get('transversal_meas_indices', []),
                ancilla_ec_instance_id=v.get('ancilla_ec_instance_id', -1),
            )
            metadata.outer_syndrome_values.append(syndrome)
        
        # Reconstruct final measurements
        for k, v in data.get('final_data_measurements', {}).items():
            measurement = FinalDataMeasurement(
                block_id=v['block_id'],
                transversal_meas_indices=v.get('transversal_meas_indices', []),
                measurement_basis=v.get('measurement_basis', 'Z'),
            )
            metadata.final_data_measurements[int(k)] = measurement
        
        return metadata
