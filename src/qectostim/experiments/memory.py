"""CSS memory experiment: generalized circuit generation for all CSS codes."""

from __future__ import annotations

from abc import ABC
from typing import Any, Dict, Optional, List
import numpy as np
import stim

from qectostim.codes.abstract_css import CSSCode
from qectostim.noise.models import NoiseModel
from .experiment import Experiment


class CSSMemoryExperiment(Experiment, ABC):
    """Memory experiment for CSS codes with unified circuit generation.
    
    This handles both geometric (Rotated Surface) and matrix-based ([[4,2,2]]) codes
    through a generic to_stim() implementation.
    """

    def __init__(
        self,
        code: CSSCode,
        noise_model: Optional[NoiseModel] = None,
        rounds: int = 1,
        basis: str = "Z",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Initialize CSS memory experiment.
        
        Args:
            code: A CSS code instance
            noise_model: Noise model to apply
            rounds: Number of measurement rounds
            basis: "Z" for Z-basis memory experiment
            metadata: Additional metadata
        """
        super().__init__(code=code, noise_model=noise_model, metadata=metadata)
        self.rounds = rounds
        self.basis = basis.upper()

    def to_stim(self) -> stim.Circuit:
        """Build Stim circuit for CSS memory experiment."""
        c = stim.Circuit()
        
        code = self.code
        n = code.n
        hx = code.hx
        hz = code.hz
        
        n_x = hx.shape[0]
        n_z = hz.shape[0]
        
        data_qubits = list(range(n))
        anc_x = list(range(n, n + n_x))
        anc_z = list(range(n + n_x, n + n_x + n_z))
        
        # Metadata
        meta = getattr(code, "metadata", {}) if hasattr(code, "metadata") else {}
        data_coords = meta.get("data_coords")
        x_stab_coords = meta.get("x_stab_coords")
        z_stab_coords = meta.get("z_stab_coords")
        x_schedule = meta.get("x_schedule")
        z_schedule = meta.get("z_schedule")
        
        # QUBIT_COORDS for data
        if data_coords is not None:
            for q, (x, y) in enumerate(data_coords):
                c.append("QUBIT_COORDS", [q], [float(x), float(y)])
        
        # QUBIT_COORDS for ancillas with distinct offsets
        if x_stab_coords is not None:
            for idx, (x, y) in enumerate(x_stab_coords):
                if idx < len(anc_x):
                    a = anc_x[idx]
                    c.append("QUBIT_COORDS", [a], [float(x) + idx * 0.05, float(y)])
        
        if z_stab_coords is not None:
            for idx, (x, y) in enumerate(z_stab_coords):
                if idx < len(anc_z):
                    a = anc_z[idx]
                    c.append("QUBIT_COORDS", [a], [float(x), float(y) + idx * 0.05])
        
        # Reset
        total_qubits = n + n_x + n_z
        if total_qubits:
            c.append("R", range(total_qubits))
        c.append("TICK")
        
        # Measurement tracking
        m_index = 0
        last_x_meas: List[Optional[int]] = [None] * n_x
        last_z_meas: List[Optional[int]] = [None] * n_z
        
        def add_detector(coord: tuple, rec_indices: List[int], t: float = 0.0) -> None:
            if not rec_indices:
                return
            nonlocal m_index
            lookbacks = [idx - m_index for idx in rec_indices]
            c.append(
                "DETECTOR",
                [stim.target_rec(lb) for lb in lookbacks],
                [float(coord[0]), float(coord[1]), t],
            )
        
        def stab_coord(coords: Optional[List], idx: int) -> tuple:
            if coords is None or idx >= len(coords):
                return (0.0, 0.0)
            x, y = coords[idx]
            return (float(x), float(y))
        
        use_geo_x = bool(x_schedule is not None and len(x_schedule) > 0)
        use_geo_z = bool(z_schedule is not None and len(z_schedule) > 0)
        
        # Rounds
        for round_num in range(self.rounds):
            t = float(round_num)
            
            # X stabilizers
            if n_x > 0:
                c.append("H", anc_x)
                
                if use_geo_x:
                    for shift_x, shift_y in x_schedule:
                        for q in data_qubits:
                            qx, qy = data_coords[q] if data_coords else (0, 0)
                            for s_idx, (sx, sy) in enumerate(x_stab_coords or []):
                                if abs((qx + shift_x) - sx) < 0.1 and abs((qy + shift_y) - sy) < 0.1:
                                    c.append("CX", [q, anc_x[s_idx]])
                else:
                    for s_idx in range(n_x):
                        for q in data_qubits:
                            if hx[s_idx, q]:
                                c.append("CX", [q, anc_x[s_idx]])
                
                c.append("H", anc_x)
                c.append("MR", anc_x)
                
                for s_idx in range(n_x):
                    if last_x_meas[s_idx] is not None:
                        add_detector(
                            stab_coord(x_stab_coords, s_idx),
                            [m_index - n_x + s_idx, last_x_meas[s_idx]],
                            t,
                        )
                    last_x_meas[s_idx] = m_index - n_x + s_idx
                
                m_index += n_x
            
            # Z stabilizers
            if n_z > 0:
                if use_geo_z:
                    for shift_x, shift_y in z_schedule:
                        for q in data_qubits:
                            qx, qy = data_coords[q] if data_coords else (0, 0)
                            for s_idx, (sx, sy) in enumerate(z_stab_coords or []):
                                if abs((qx + shift_x) - sx) < 0.1 and abs((qy + shift_y) - sy) < 0.1:
                                    c.append("CX", [anc_z[s_idx], q])
                else:
                    for s_idx in range(n_z):
                        for q in data_qubits:
                            if hz[s_idx, q]:
                                c.append("CX", [anc_z[s_idx], q])
                
                c.append("MR", anc_z)
                
                for s_idx in range(n_z):
                    if last_z_meas[s_idx] is not None:
                        add_detector(
                            stab_coord(z_stab_coords, s_idx),
                            [m_index - n_z + s_idx, last_z_meas[s_idx]],
                            t,
                        )
                    last_z_meas[s_idx] = m_index - n_z + s_idx
                
                m_index += n_z
            
            c.append("SHIFT_COORDS", [], [0, 0, 1])
        
        # Final measurement
        if n > 0:
            c.append("M", data_qubits)
            
            for s_idx in range(n_x):
                recs = []
                for q in data_qubits:
                    if hx[s_idx, q]:
                        recs.append(m_index - n + q)
                if last_x_meas[s_idx] is not None:
                    recs.append(last_x_meas[s_idx])
                add_detector(
                    stab_coord(x_stab_coords, s_idx),
                    recs,
                    t=float(self.rounds),
                )
            
            for s_idx in range(n_z):
                recs = []
                for q in data_qubits:
                    if hz[s_idx, q]:
                        recs.append(m_index - n + q)
                if last_z_meas[s_idx] is not None:
                    recs.append(last_z_meas[s_idx])
                add_detector(
                    stab_coord(z_stab_coords, s_idx),
                    recs,
                    t=float(self.rounds),
                )
            
            m_index += n
        
        # Observables
        logical_z_ops = code.logical_z_ops if hasattr(code, "logical_z_ops") else []
        
        # m_index now points past all measurements, so we need to count backward
        # Final measurement of data qubits starts at m_index - n
        for logical_idx, op in enumerate(logical_z_ops):
            recs = []
            for q, pauli in enumerate(op):
                if pauli in ["Z", "Y"]:
                    # rec[-1] is the last measurement (qubit n-1 of data)
                    # rec[-n] is the first measurement (qubit 0 of data)
                    rec_lookback = n - q
                    recs.append(stim.target_rec(-rec_lookback))
            if recs:
                c.append("OBSERVABLE_INCLUDE", recs, [float(logical_idx)])
        
        return c
