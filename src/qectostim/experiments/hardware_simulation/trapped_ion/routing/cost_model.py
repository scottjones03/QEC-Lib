# src/qectostim/experiments/hardware_simulation/trapped_ion/routing/cost_model.py
"""
WISE Cost Model for routing optimization.

This module provides the WISECostModel class that estimates costs
for transport operations in WISE trapped-ion architectures.
"""

from __future__ import annotations

from collections import defaultdict
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from qectostim.experiments.hardware_simulation.trapped_ion.physics import (
    DEFAULT_CALIBRATION as _CAL,
)


class WISECostModel:
    """Cost model for WISE trapped-ion routing.
    
    Estimates costs based on:
    - Transport time (proportional to distance)
    - Gate execution time
    - Reconfiguration overhead
    
    This implements the CostModel interface from core/compiler.py.
    """
    
    def __init__(
        self,
        transport_time_per_unit: float = _CAL.shuttle_time * 1e6,  # μs per grid unit
        gate_time_2q: float = _CAL.ms_gate_time * 1e6,             # μs for two-qubit gate
        gate_time_1q: float = _CAL.single_qubit_gate_time * 1e6,   # μs for single-qubit gate
        reconfiguration_overhead: float = _CAL.junction_time * 1e6,  # μs per reconfiguration
    ):
        self.transport_time_per_unit = transport_time_per_unit
        self.gate_time_2q = gate_time_2q
        self.gate_time_1q = gate_time_1q
        self.reconfiguration_overhead = reconfiguration_overhead
    
    def operation_cost(
        self,
        operation_type: str,
        qubits: Tuple[int, ...],
        params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Estimate cost of a single operation."""
        params = params or {}
        
        if operation_type in ("H_SWAP", "V_SWAP", "transport", "swap"):
            distance = params.get("distance", 1.0)
            return distance * self.transport_time_per_unit
        elif operation_type in ("MS", "XX", "ZZ", "2Q", "gate_2q"):
            return self.gate_time_2q
        elif operation_type in ("R", "RZ", "RX", "RY", "1Q", "gate_1q"):
            return self.gate_time_1q
        elif operation_type == "reconfiguration":
            return self.reconfiguration_overhead
        else:
            return 1.0  # Unknown operation
    
    def sequence_cost(
        self,
        operations: List[Tuple[str, Tuple[int, ...], Optional[Dict[str, Any]]]],
    ) -> float:
        """Estimate total cost of an operation sequence.
        
        Accounts for parallelism within phases (H or V).
        """
        if not operations:
            return 0.0
        
        # Group by phase for parallelism
        phase_costs: Dict[str, float] = defaultdict(float)
        
        for op_type, qubits, params in operations:
            cost = self.operation_cost(op_type, qubits, params)
            
            if op_type == "H_SWAP":
                phase_costs["H"] = max(phase_costs["H"], cost)
            elif op_type == "V_SWAP":
                phase_costs["V"] = max(phase_costs["V"], cost)
            else:
                phase_costs["other"] += cost
        
        return sum(phase_costs.values())
    
    def compare(self, cost_a: float, cost_b: float) -> int:
        """Compare two costs (-1 if a better, 0 if equal, 1 if b better)."""
        if cost_a < cost_b:
            return -1
        elif cost_a > cost_b:
            return 1
        return 0
    
    def is_acceptable(self, cost: float, threshold: float = 10000.0) -> bool:
        """Check if a cost is acceptable."""
        return cost < threshold


__all__ = [
    "WISECostModel",
]
