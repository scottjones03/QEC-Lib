# src/qectostim/experiments/logical_gates.py
"""
Logical Gate Experiment with Automatic Gadget Routing.

This module provides an experiment class that:
1. Selects the best gadget implementation for a logical gate
2. Runs the gate using the FaultTolerantGadgetExperiment pattern
3. Returns the logical error rate

The routing logic considers:
- Available transversal gates for the code
- Teleportation-based alternatives
- Surgery operations for multi-qubit gates

Example usage:
    >>> from qectostim.experiments.logical_gates import LogicalGateExperiment
    >>> from qectostim.codes.surface import RotatedSurfaceCode
    >>> from qectostim.noise.models import CircuitDepolarizingNoise
    >>>
    >>> code = RotatedSurfaceCode(distance=3)
    >>> noise = CircuitDepolarizingNoise(p1=0.001, p2=0.001)
    >>> 
    >>> exp = LogicalGateExperiment(
    ...     codes=[code],
    ...     gate_name="H",
    ...     noise_model=noise,
    ... )
    >>> circuit = exp.to_stim()
"""
from __future__ import annotations
from typing import Any, Dict, Optional, List, Type
from dataclasses import dataclass

import stim

from qectostim.experiments.experiment import Experiment
from qectostim.codes.abstract_code import Code
from qectostim.noise.models import NoiseModel
from qectostim.gadgets.base import Gadget


@dataclass
class GateRoute:
    """
    Description of how to implement a logical gate.
    
    Attributes:
        gate_name: Name of the gate (H, S, T, CNOT, etc.)
        gadget_class: Gadget class to use
        gadget_type: Type of implementation (transversal, teleportation, surgery)
        requires_codes: Number of code blocks required
        kwargs: Additional arguments for gadget constructor
    """
    gate_name: str
    gadget_class: Type[Gadget]
    gadget_type: str
    requires_codes: int = 1
    kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


class GateRouter:
    """
    Routes logical gates to appropriate gadget implementations.
    
    The router considers:
    1. Whether the gate is transversal for the given code(s)
    2. Teleportation-based alternatives for Clifford gates
    3. Surgery for multi-qubit gates
    
    Priority order:
    1. Transversal (if available) - fastest, lowest overhead
    2. Teleportation (for Cliffords) - works on any CSS code
    3. Surgery (for two-qubit gates) - universal but highest overhead
    """
    
    def __init__(self):
        # Build gate routing table
        self._routes: Dict[str, List[GateRoute]] = {}
        self._build_routing_table()
    
    def _build_routing_table(self) -> None:
        """Build the gate routing table."""
        from qectostim.gadgets.bin.transversal import (
            TransversalHadamard,
            TransversalS,
            TransversalSDag,
            TransversalT,
            TransversalTDag,
            TransversalX,
            TransversalY,
            TransversalZ,
            TransversalCNOT,
            TransversalCZ,
        )
        from qectostim.gadgets.bin.teleportation import (
            TeleportedHadamard,
            TeleportedS,
            TeleportedSDag,
            TeleportedT,
        )
        from qectostim.gadgets.bin.css_surgery import SurgeryCNOT
        
        # Single-qubit gates
        self._routes["H"] = [
            GateRoute("H", TransversalHadamard, "transversal"),
            GateRoute("H", TeleportedHadamard, "teleportation"),
        ]
        self._routes["S"] = [
            GateRoute("S", TransversalS, "transversal"),
            GateRoute("S", TeleportedS, "teleportation"),
        ]
        self._routes["S_DAG"] = [
            GateRoute("S_DAG", TransversalSDag, "transversal"),
            GateRoute("S_DAG", TeleportedSDag, "teleportation"),
        ]
        self._routes["T"] = [
            GateRoute("T", TransversalT, "transversal"),
            GateRoute("T", TeleportedT, "teleportation"),
        ]
        self._routes["T_DAG"] = [
            GateRoute("T_DAG", TransversalTDag, "transversal"),
        ]
        self._routes["X"] = [
            GateRoute("X", TransversalX, "transversal"),
        ]
        self._routes["Y"] = [
            GateRoute("Y", TransversalY, "transversal"),
        ]
        self._routes["Z"] = [
            GateRoute("Z", TransversalZ, "transversal"),
        ]
        
        # Two-qubit gates
        self._routes["CNOT"] = [
            GateRoute("CNOT", TransversalCNOT, "transversal", requires_codes=2),
            GateRoute("CNOT", SurgeryCNOT, "surgery", requires_codes=2),
        ]
        self._routes["CX"] = self._routes["CNOT"]  # Alias
        self._routes["CZ"] = [
            GateRoute("CZ", TransversalCZ, "transversal", requires_codes=2),
        ]
    
    def route(
        self,
        gate_name: str,
        codes: List[Code],
        prefer_transversal: bool = True,
    ) -> Optional[GateRoute]:
        """
        Find the best route for a gate on given codes.
        
        Parameters
        ----------
        gate_name : str
            Name of the logical gate.
        codes : List[Code]
            Code(s) to apply gate to.
        prefer_transversal : bool
            If True, prefer transversal over teleportation.
            
        Returns
        -------
        Optional[GateRoute]
            Best route, or None if no route found.
        """
        gate_key = gate_name.upper()
        
        if gate_key not in self._routes:
            return None
        
        routes = self._routes[gate_key]
        
        for route in routes:
            # Check if we have enough codes
            if len(codes) < route.requires_codes:
                continue
            
            # Check transversal support
            if route.gadget_type == "transversal" and prefer_transversal:
                # Check if transversal is supported
                if self._check_transversal_support(codes, gate_name):
                    return route
            
            # Teleportation works on any CSS code
            elif route.gadget_type == "teleportation":
                if all(c.is_css for c in codes):
                    return route
            
            # Surgery works on any CSS code pair
            elif route.gadget_type == "surgery":
                if all(c.is_css for c in codes):
                    return route
        
        # Fallback: return first compatible route
        for route in routes:
            if len(codes) >= route.requires_codes:
                return route
        
        return None
    
    def _check_transversal_support(
        self,
        codes: List[Code],
        gate_name: str,
    ) -> bool:
        """Check if transversal gate is supported by all codes."""
        for code in codes:
            supported = code.transversal_gates()
            if gate_name.upper() not in [g.upper() for g in supported]:
                return False
        return True


# Global router instance (lazy initialization to avoid circular imports)
_router: Optional[GateRouter] = None


def _get_router() -> GateRouter:
    """Get or create the global router instance."""
    global _router
    if _router is None:
        _router = GateRouter()
    return _router


def get_gate_route(
    gate_name: str,
    codes: List[Code],
) -> Optional[GateRoute]:
    """Get the best route for a logical gate."""
    return _get_router().route(gate_name, codes)


class LogicalGateExperiment(Experiment):
    """
    Experiment that runs a logical gate with automatic gadget selection.
    
    This class:
    1. Routes the gate to the best available gadget implementation
    2. Creates a FaultTolerantGadgetExperiment with the selected gadget
    3. Generates the complete Stim circuit
    
    Parameters
    ----------
    codes : List[Code]
        Code(s) to apply gate to.
    gate_name : str
        Name of the logical gate (H, S, T, CNOT, etc.)
    noise_model : NoiseModel
        Noise model for the experiment.
    num_rounds_before : int
        Stabilizer rounds before gate.
    num_rounds_after : int
        Stabilizer rounds after gate.
    gadget_override : Optional[Gadget]
        If provided, use this gadget instead of routing.
    metadata : Optional[Dict]
        Additional experiment metadata.
    """
    
    def __init__(
        self,
        codes: List[Code],
        gate_name: str,
        noise_model: NoiseModel,
        num_rounds_before: int = 3,
        num_rounds_after: int = 3,
        gadget_override: Optional[Gadget] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(codes[0], noise_model, metadata)
        
        self.codes = codes
        self.gate_name = gate_name
        self.num_rounds_before = num_rounds_before
        self.num_rounds_after = num_rounds_after
        self.gadget_override = gadget_override
        
        # Route to gadget if not overridden
        self._route: Optional[GateRoute] = None
        self._gadget: Optional[Gadget] = None
        self._resolve_gadget()
    
    def _resolve_gadget(self) -> None:
        """Resolve which gadget to use."""
        if self.gadget_override is not None:
            self._gadget = self.gadget_override
            return
        
        # Route the gate
        self._route = get_gate_route(self.gate_name, self.codes)
        
        if self._route is None:
            raise ValueError(
                f"No gadget implementation found for gate '{self.gate_name}' "
                f"with {len(self.codes)} codes"
            )
        
        # Create gadget instance
        self._gadget = self._route.gadget_class(**self._route.kwargs)
    
    def get_gadget(self) -> Gadget:
        """Get the selected gadget."""
        if self._gadget is None:
            raise RuntimeError("Gadget not resolved")
        return self._gadget
    
    def get_route_info(self) -> Dict[str, Any]:
        """Get information about the selected route."""
        if self._route is None and self.gadget_override is not None:
            return {
                "gate_name": self.gate_name,
                "gadget_class": type(self.gadget_override).__name__,
                "gadget_type": "override",
            }
        elif self._route is not None:
            return {
                "gate_name": self._route.gate_name,
                "gadget_class": self._route.gadget_class.__name__,
                "gadget_type": self._route.gadget_type,
                "requires_codes": self._route.requires_codes,
            }
        return {}
    
    def to_stim(self) -> stim.Circuit:
        """
        Generate the Stim circuit for the logical gate experiment.
        
        Returns
        -------
        stim.Circuit
            Complete circuit with detectors and observables.
        """
        from qectostim.experiments.ft_gadget_experiment import (
            FaultTolerantGadgetExperiment,
        )
        
        # Create FT gadget experiment
        ft_exp = FaultTolerantGadgetExperiment(
            codes=self.codes,
            gadget=self._gadget,
            noise_model=self.noise_model,
            num_rounds_before=self.num_rounds_before,
            num_rounds_after=self.num_rounds_after,
            metadata=self._metadata,
        )
        
        return ft_exp.to_stim()
    
    def run_decode(
        self,
        decoder_name: str = "pymatching",
        num_shots: int = 10000,
        **kwargs,
    ):
        """
        Run the experiment and decode results.
        
        Parameters
        ----------
        decoder_name : str
            Decoder to use.
        num_shots : int
            Number of shots.
        **kwargs
            Additional decoder arguments.
            
        Returns
        -------
        FTGadgetExperimentResult
            Experiment results.
        """
        from qectostim.experiments.ft_gadget_experiment import (
            FaultTolerantGadgetExperiment,
        )
        
        ft_exp = FaultTolerantGadgetExperiment(
            codes=self.codes,
            gadget=self._gadget,
            noise_model=self.noise_model,
            num_rounds_before=self.num_rounds_before,
            num_rounds_after=self.num_rounds_after,
            metadata=self._metadata,
        )
        
        result = ft_exp.run_decode(
            decoder_name=decoder_name,
            num_shots=num_shots,
            **kwargs,
        )
        
        # Add routing info to result
        result.extra["route_info"] = self.get_route_info()
        
        return result


def run_logical_gate(
    codes: List[Code],
    gate_name: str,
    noise_model: NoiseModel,
    num_rounds: int = 3,
    decoder_name: str = "pymatching",
    num_shots: int = 10000,
):
    """
    Convenience function to run a logical gate experiment.
    
    Parameters
    ----------
    codes : List[Code]
        Code(s) to apply gate to.
    gate_name : str
        Name of the logical gate.
    noise_model : NoiseModel
        Noise model.
    num_rounds : int
        Stabilizer rounds before and after.
    decoder_name : str
        Decoder to use.
    num_shots : int
        Number of shots.
        
    Returns
    -------
    FTGadgetExperimentResult
        Experiment results.
    """
    exp = LogicalGateExperiment(
        codes=codes,
        gate_name=gate_name,
        noise_model=noise_model,
        num_rounds_before=num_rounds,
        num_rounds_after=num_rounds,
    )
    return exp.run_decode(decoder_name=decoder_name, num_shots=num_shots)
