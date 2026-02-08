"""
Stabilizer Scheduling Module for Deterministic Detector Emission.

═══════════════════════════════════════════════════════════════════════════════
OVERVIEW
═══════════════════════════════════════════════════════════════════════════════

This module provides scheduling logic to maximize deterministic detector coverage.
The key insight is that stabilizer measurement ORDER determines which detectors
can be made deterministic:

**ANCHOR DETECTORS** (after preparation):
- If preparing in basis B (e.g., |0⟩ = Z basis, |+⟩ = X basis):
- B-type stabilizers measured FIRST can have anchor detectors
- Because B-stabilizers are deterministic immediately after B-basis prep

**BOUNDARY DETECTORS** (before destructive measurement):  
- If measuring in basis B:
- B-type stabilizers measured LAST can have boundary detectors
- Because B-stabilizer syndrome correlates with final B-basis data measurement

═══════════════════════════════════════════════════════════════════════════════
EXAMPLE: |0⟩ preparation, Z-basis measurement
═══════════════════════════════════════════════════════════════════════════════

Preparation |0⟩ → Z stabilizers are eigenvalue +1 (deterministic)
First round: Measure Z first, X second
  → Z anchor detectors compare first Z syndrome to deterministic 0

Final measurement MZ → Z stabilizers correlate with data Z products
Last round: Measure X first, Z last  
  → Z boundary detectors compare last Z syndrome to MZ data products

═══════════════════════════════════════════════════════════════════════════════
GATE OPTIMIZATION
═══════════════════════════════════════════════════════════════════════════════

We prefer RX/MX over H gates where possible:
- |+⟩ preparation: Use RX instead of R + H
- X-basis measurement: Use MX instead of H + M
- This avoids extra noisy H gates

═══════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class StabilizerBasis(Enum):
    """
    Stabilizer basis types.
    
    Values match qectostim.experiments.stabilizer_rounds.base.StabilizerBasis
    to allow seamless comparison and usage.
    """
    X = "x"
    Z = "z"
    BOTH = "both"


@dataclass
class RoundSchedule:
    """
    Scheduling decision for a single round of stabilizer measurements.
    
    Attributes
    ----------
    first_basis : StabilizerBasis
        Which stabilizer type to measure first in this round.
    last_basis : StabilizerBasis
        Which stabilizer type to measure last in this round.
    is_anchor_round : bool
        Whether this is the first round (anchor detectors possible).
    is_boundary_round : bool
        Whether this is the last round (boundary detectors possible).
    anchor_basis : Optional[StabilizerBasis]
        Which basis has deterministic anchor detectors (if anchor round).
    boundary_basis : Optional[StabilizerBasis]
        Which basis has deterministic boundary detectors (if boundary round).
    """
    first_basis: StabilizerBasis
    last_basis: StabilizerBasis
    is_anchor_round: bool = False
    is_boundary_round: bool = False
    anchor_basis: Optional[StabilizerBasis] = None
    boundary_basis: Optional[StabilizerBasis] = None


@dataclass
class BlockSchedule:
    """
    Complete scheduling for a code block across all rounds.
    
    Attributes
    ----------
    block_name : str
        Name of the code block.
    prep_basis : str
        Preparation basis: "0" for |0⟩, "+" for |+⟩.
    meas_basis : str
        Final measurement basis: "Z" or "X".
    round_schedules : List[RoundSchedule]
        Per-round scheduling decisions.
    use_rx_prep : bool
        Whether to use RX instead of R+H for |+⟩ prep.
    use_mx_meas : bool
        Whether to use MX instead of H+M for X-basis meas.
    """
    block_name: str
    prep_basis: str
    meas_basis: str
    round_schedules: List[RoundSchedule] = field(default_factory=list)
    use_rx_prep: bool = True      # Prefer RX over R+H
    use_mx_meas: bool = True      # Prefer MX over H+M


class StabilizerScheduler:
    """
    Scheduler for stabilizer measurements to maximize deterministic detectors.
    
    This class computes optimal scheduling based on:
    - Preparation basis (determines anchor detector determinism)
    - Measurement basis (determines boundary detector determinism)
    - Gate transformations (determines crossing detector structure)
    
    Usage:
        scheduler = StabilizerScheduler()
        schedule = scheduler.compute_schedule(
            prep_basis="0",
            meas_basis="Z", 
            num_rounds=5,
        )
        # schedule.first_round_basis -> StabilizerBasis.Z (for anchors)
        # schedule.last_round_basis -> StabilizerBasis.Z (for boundaries)
    """
    
    def __init__(self):
        """Initialize the scheduler."""
        pass
    
    def get_anchor_deterministic_basis(
        self,
        prep_basis: str,
    ) -> StabilizerBasis:
        """
        Determine which stabilizer basis is deterministic after preparation.
        
        |0⟩ preparation: Z stabilizers are +1 eigenvalue (deterministic)
        |+⟩ preparation: X stabilizers are +1 eigenvalue (deterministic)
        
        Parameters
        ----------
        prep_basis : str
            Preparation basis: "0" for |0⟩, "+" for |+⟩.
            
        Returns
        -------
        StabilizerBasis
            The basis whose stabilizers are deterministic after prep.
        """
        if prep_basis == "0":
            return StabilizerBasis.Z
        elif prep_basis == "+":
            return StabilizerBasis.X
        else:
            raise ValueError(f"Unknown prep basis: {prep_basis}")
    
    def get_boundary_deterministic_basis(
        self,
        meas_basis: str,
    ) -> StabilizerBasis:
        """
        Determine which stabilizer basis has deterministic boundary detectors.
        
        MZ measurement: Z stabilizers correlate with MZ products (deterministic)
        MX measurement: X stabilizers correlate with MX products (deterministic)
        
        Parameters
        ----------
        meas_basis : str
            Measurement basis: "Z" for MZ, "X" for MX.
            
        Returns
        -------
        StabilizerBasis
            The basis whose boundary detectors are deterministic.
        """
        if meas_basis.upper() == "Z":
            return StabilizerBasis.Z
        elif meas_basis.upper() == "X":
            return StabilizerBasis.X
        else:
            raise ValueError(f"Unknown meas basis: {meas_basis}")
    
    def compute_block_schedule(
        self,
        block_name: str,
        prep_basis: str,
        meas_basis: str,
        num_rounds: int,
        default_ordering: Optional[str] = None,
    ) -> BlockSchedule:
        """
        Compute complete scheduling for a single code block.
        
        Parameters
        ----------
        block_name : str
            Name of the code block.
        prep_basis : str
            Preparation basis: "0" or "+".
        meas_basis : str
            Final measurement basis: "Z" or "X".
        num_rounds : int
            Total number of syndrome measurement rounds.
        default_ordering : Optional[str]
            Override for middle-round ordering: "Z_FIRST" or "X_FIRST".
            When None, uses Z-first as default for middle rounds.
            
        Returns
        -------
        BlockSchedule
            Complete scheduling for the block.
        """
        anchor_basis = self.get_anchor_deterministic_basis(prep_basis)
        boundary_basis = self.get_boundary_deterministic_basis(meas_basis)
        
        # Determine prep/meas gate optimization
        use_rx = (prep_basis == "+")
        use_mx = (meas_basis.upper() == "X")
        
        round_schedules = []
        for r in range(num_rounds):
            is_first = (r == 0)
            is_last = (r == num_rounds - 1)
            
            if is_first and is_last:
                # Single round: anchor takes priority, but try to accommodate both
                # Anchor needs deterministic first; boundary needs deterministic last
                # If same basis, easy: that basis first AND last (measure once)
                # If different, anchor wins (measure anchor basis first)
                if anchor_basis == boundary_basis:
                    first = anchor_basis
                    last = anchor_basis
                else:
                    # Conflict: anchor wins for first round
                    first = anchor_basis
                    other = StabilizerBasis.X if anchor_basis == StabilizerBasis.Z else StabilizerBasis.Z
                    last = other
            elif is_first:
                # First round: anchor deterministic basis first
                first = anchor_basis
                other = StabilizerBasis.X if anchor_basis == StabilizerBasis.Z else StabilizerBasis.Z
                last = other
            elif is_last:
                # Last round: boundary deterministic basis last
                other = StabilizerBasis.X if boundary_basis == StabilizerBasis.Z else StabilizerBasis.Z
                first = other
                last = boundary_basis
            else:
                # Middle rounds: use gadget's requested ordering or default (Z first, X second)
                if default_ordering == "X_FIRST":
                    first = StabilizerBasis.X
                    last = StabilizerBasis.Z
                else:
                    # Default: Z first, X second
                    first = StabilizerBasis.Z
                    last = StabilizerBasis.X
            
            schedule = RoundSchedule(
                first_basis=first,
                last_basis=last,
                is_anchor_round=is_first,
                is_boundary_round=is_last,
                anchor_basis=anchor_basis if is_first else None,
                boundary_basis=boundary_basis if is_last else None,
            )
            round_schedules.append(schedule)
        
        return BlockSchedule(
            block_name=block_name,
            prep_basis=prep_basis,
            meas_basis=meas_basis.upper(),
            round_schedules=round_schedules,
            use_rx_prep=use_rx,
            use_mx_meas=use_mx,
        )
    
    def get_reset_instruction(self, basis: str) -> str:
        """
        Get the Stim instruction for reset in given basis.
        
        Parameters
        ----------
        basis : str
            "Z" for |0⟩, "X" for |+⟩.
            
        Returns
        -------
        str
            "R" for Z-basis, "RX" for X-basis.
        """
        if basis.upper() == "Z":
            return "R"
        elif basis.upper() == "X":
            return "RX"
        else:
            raise ValueError(f"Unknown basis: {basis}")
    
    def get_meas_reset_instruction(self, basis: str) -> str:
        """
        Get the Stim instruction for measure-and-reset in given basis.
        
        Parameters
        ----------
        basis : str
            "Z" for computational, "X" for Hadamard.
            
        Returns
        -------
        str
            "MR" for Z-basis, "MRX" for X-basis.
        """
        if basis.upper() == "Z":
            return "MR"
        elif basis.upper() == "X":
            return "MRX"
        else:
            raise ValueError(f"Unknown basis: {basis}")



