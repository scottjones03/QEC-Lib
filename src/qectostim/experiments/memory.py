from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional

import numpy as np
import stim

from qectostim.codes.abstract_code import Code
from qectostim.codes.abstract_css import CSSCode
from qectostim.experiments.experiment import Experiment
from qectostim.noise.models import NoiseModel


class MemoryExperiment(Experiment):
    """
    Repeated stabilizer measurement (memory) experiment for a single logical qubit.
    """

    def __init__(
        self,
        code: Code,
        noise_model: NoiseModel | None,
        rounds: int,
        logical_qubit: int = 0,
        initial_state: str = "0",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(code, noise_model, metadata)
        self.rounds = rounds
        self.logical_qubit = logical_qubit
        self.initial_state = initial_state
        self.operation = "memory"

    @abc.abstractmethod
    def to_stim(self) -> stim.Circuit:
        ...


class CSSMemoryExperiment(MemoryExperiment):
    """
    Simple memory experiment for arbitrary CSS codes (placeholder implementation).
    """

    def __init__(
        self,
        code: CSSCode,
        rounds: int,
        noise_model: Dict[str, Any] | None = None,
        basis: str = "Z",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(code, noise_model, rounds=rounds, metadata=metadata)
        self.basis = basis  # "Z" or "X"

    def to_stim(self) -> stim.Circuit:
        """
        Build a generic CSS memory experiment with detectors.

        If the code exposes geometric metadata:
          - metadata["data_coords"]       : list of (x, y) for data qubits
          - metadata["x_stab_coords"]    : list of (x, y) for X-check ancillas
          - metadata["z_stab_coords"]    : list of (x, y) for Z-check ancillas
          - metadata["x_schedule"]       : list of (dx, dy) steps
          - metadata["z_schedule"]       : list of (dx, dy) steps

        then we respect that schedule (e.g. rotated surface code with 4 CNOT
        “phases” per stabilizer, moving clockwise).

        Otherwise, we fall back to a naive (but consistent) CSS stabilizer
        circuit:
          - X checks: H on ancilla, CNOT(data->ancilla), H, then measure ancilla.
          - Z checks: CNOT(data->ancilla), then measure ancilla.

        Pattern:
          1. Reset data + ancillas.
          2. Repeat `rounds` times:
             - run X and Z checks,
             - measure ancillas,
             - time-like DETECTORs comparing each ancilla's current vs previous meas.
          3. Measure data once at the end.
          4. Space-like DETECTORs combining last Z-syndrome with data parity.
          5. Single OBSERVABLE_INCLUDE for the logical observable.
        """
        code = self.code
        n = code.n
        hx = code.hx
        hz = code.hz

        # Cache basis as upper-case once.
        basis = self.basis.upper()

        # --- Align layer matrices (hx/hz) with provided geometric coords.
        # Some chain-complex constructions may order/label faces differently,
        # so ensure that the X layer count matches x_stab_coords and likewise for Z.
        meta = getattr(code, "metadata", {}) if hasattr(code, "metadata") else {}
        x_coords_meta = meta.get("x_stab_coords")
        z_coords_meta = meta.get("z_stab_coords")
        if x_coords_meta is not None and z_coords_meta is not None:
            if hx.shape[0] != len(x_coords_meta) and hz.shape[0] == len(x_coords_meta):
                # Swap layers so hx corresponds to X faces and hz to Z faces.
                hx, hz = hz, hx

        n_x = hx.shape[0]
        n_z = hz.shape[0]

        data_qubits = list(range(n))
        anc_x = list(range(n, n + n_x))
        anc_z = list(range(n + n_x, n + n_x + n_z))

        c = stim.Circuit()

        # ---- Geometry / metadata ------------------------------------------
        meta = getattr(code, "metadata", {}) if hasattr(code, "metadata") else {}
        data_coords = meta.get("data_coords")
        x_stab_coords = meta.get("x_stab_coords")
        z_stab_coords = meta.get("z_stab_coords")
        x_schedule = meta.get("x_schedule")
        z_schedule = meta.get("z_schedule")

        coord_to_data: dict[tuple[float, float], int] = {}

        if data_coords is not None:
            for q, (x, y) in enumerate(data_coords):
                coord = (float(x), float(y))
                if coord in coord_to_data:
                    continue
                coord_to_data[coord] = q
                c.append("QUBIT_COORDS", [q], [coord[0], coord[1]])

        if x_stab_coords is not None:
            for a, (x, y) in zip(anc_x, x_stab_coords):
                c.append("QUBIT_COORDS", [a], [float(x), float(y)])

        if z_stab_coords is not None:
            for a, (x, y) in zip(anc_z, z_stab_coords):
                c.append("QUBIT_COORDS", [a], [float(x), float(y)])

        # ---- Initial preparation ------------------------------------------
        total_qubits = n + n_x + n_z
        if total_qubits:
            c.append("R", range(total_qubits))

        # Prepare logical in chosen basis (crude: apply H to all data for X-basis).
        if self.basis.upper() == "X" and n > 0:
            c.append("H", data_qubits)

        c.append("TICK")

        # Measurement index bookkeeping.
        m_index = 0  # total number of measurement results so far
        last_x_meas: list[Optional[int]] = [None] * n_x
        last_z_meas: list[Optional[int]] = [None] * n_z

        def add_detector(coord: tuple[float, float],
                         rec_indices: list[int],
                         t: float = 0.0) -> None:
            """Emit a DETECTOR at space-time coord with rec lookbacks.

            `rec_indices` are absolute measurement indices (0,1,2,...).
            At the moment we call this, `m_index` is the total #meas so far.
            Stim wants lookbacks (negative indices), so we convert via:
                lookback = idx - m_index   (which is <= -1)
            """
            if not rec_indices:
                return
            lookbacks = [idx - m_index for idx in rec_indices]
            c.append(
                "DETECTOR",
                [stim.target_rec(lb) for lb in lookbacks],
                [float(coord[0]), float(coord[1]), t],
            )

        def stab_coord(coords: Optional[List[tuple[float, float]]], idx: int) -> tuple[float, float]:
            if coords is None or idx >= len(coords):
                return (0.0, 0.0)
            x, y = coords[idx]
            return (float(x), float(y))

        use_geo_x = bool(
            x_schedule
            and data_coords is not None
            and x_stab_coords is not None
            and len(x_stab_coords) == n_x
        )
        use_geo_z = bool(
            z_schedule
            and data_coords is not None
            and z_stab_coords is not None
            and len(z_stab_coords) == n_z
        )

        # ---- Syndrome rounds ----------------------------------------------
        interleaved_geo = bool(
            (x_schedule and z_schedule)
            and (data_coords is not None)
            and (x_stab_coords is not None)
            and (z_stab_coords is not None)
            and (len(x_stab_coords) == n_x)
            and (len(z_stab_coords) == n_z)
        )

        for r in range(self.rounds):
            # Prepare ancillas for this round.
            # X ancillas: rotate into |+> at the start of each round.
            if n_x:
                if interleaved_geo or use_geo_x:
                    for s_idx, (sx, sy) in enumerate(x_stab_coords or []):
                        a = anc_x[s_idx]
                        c.append("H", [a])
                else:
                    for s_idx in range(n_x):
                        a = anc_x[s_idx]
                        c.append("H", [a])

            # Z ancillas start in |0> from the initial global reset or the
            # previous round's demolition measurement, so no per-round reset
            # is needed here.

            if interleaved_geo:
                # Interleave X and Z checks per phase (Stim style)
                for (dx_x, dy_x), (dx_z, dy_z) in zip(x_schedule, z_schedule):
                    c.append("TICK")
                    # X layer: CNOT(data -> ancilla)
                    for s_idx, (sx, sy) in enumerate(x_stab_coords):
                        a = anc_x[s_idx]
                        nbr = (float(sx) + dx_x, float(sy) + dy_x)
                        dq = coord_to_data.get(nbr)
                        if dq is not None:
                            c.append("CNOT", [dq, a])
                    # Z layer: CNOT(data -> ancilla)
                    for s_idx, (sx, sy) in enumerate(z_stab_coords):
                        a = anc_z[s_idx]
                        nbr = (float(sx) + dx_z, float(sy) + dy_z)
                        dq = coord_to_data.get(nbr)
                        if dq is not None:
                            c.append("CNOT", [dq, a])
                # Rotate X ancillas back before measurement
                for s_idx in range(n_x):
                    a = anc_x[s_idx]
                    c.append("H", [a])
            else:
                # Fallback: perform X layer then Z layer (non-interleaved)
                if n_x:
                    if use_geo_x:
                        for dx, dy in x_schedule or []:
                            c.append("TICK")
                            for s_idx, (sx, sy) in enumerate(x_stab_coords or []):
                                a = anc_x[s_idx]
                                nbr = (float(sx) + dx, float(sy) + dy)
                                dq = coord_to_data.get(nbr)
                                if dq is not None:
                                    c.append("CNOT", [dq, a])
                        for s_idx in range(n_x):
                            a = anc_x[s_idx]
                            c.append("H", [a])
                    else:
                        for s_idx, row in enumerate(hx):
                            a = anc_x[s_idx]
                            # H, then entangle, then H
                            # (H already applied above when preparing.)
                            for dq in np.where(row == 1)[0]:
                                c.append("CNOT", [dq, a])
                            c.append("H", [a])
                if n_z:
                    if use_geo_z:
                        for dx, dy in z_schedule or []:
                            c.append("TICK")
                            for s_idx, (sx, sy) in enumerate(z_stab_coords or []):
                                a = anc_z[s_idx]
                                nbr = (float(sx) + dx, float(sy) + dy)
                                dq = coord_to_data.get(nbr)
                                if dq is not None:
                                    c.append("CNOT", [dq, a])
                    else:
                        for s_idx, row in enumerate(hz):
                            a = anc_z[s_idx]
                            for dq in np.where(row == 1)[0]:
                                c.append("CNOT", [dq, a])

            # ---- Measure ancillas in batches (like Stim) ----
            # For a CSS memory experiment, we need BOTH X and Z syndromes:
            # - Z-basis memory: Z stabilizers detect X errors (needed for correction)
            #                   X stabilizers also detect Z errors that would flip the observable
            # - X-basis memory: X stabilizers detect Z errors (needed for correction)
            #                   Z stabilizers also detect X errors
            #
            # Both types of stabilizers should be measured in EVERY round to enable
            # proper error correction. The final round may use M instead of MR.
            
            # Collect all X-ancillas to measure in this round
            # X-ancillas are measured every round (not just for X-basis)
            x_ancillas_to_measure = []
            for s_idx in range(n_x):
                # Always measure X-ancillas in every round for proper syndrome extraction
                x_ancillas_to_measure.append((s_idx, anc_x[s_idx]))
            
            # Collect all Z-ancillas to measure in this round
            # Z-ancillas are measured every round for syndrome extraction and space-like detectors
            z_ancillas_to_measure = []
            for s_idx in range(n_z):
                z_ancillas_to_measure.append((s_idx, anc_z[s_idx]))
            
            # Measure X-ancillas in one batch (if any)
            x_meas_start_idx = m_index
            if x_ancillas_to_measure:
                x_qubits = [a for _, a in x_ancillas_to_measure]
                if r < self.rounds - 1:
                    c.append("MR", x_qubits)
                else:
                    c.append("MR", x_qubits)  # Use MR even in last round for consistency
                m_index += len(x_qubits)
            
            # Create time-like detectors for X-ancillas
            # For Z-basis, X-ancillas detect Z errors (which don't affect Z-basis logical state directly,
            # but are needed for proper decoding when errors propagate)
            for offset, (s_idx, _) in enumerate(x_ancillas_to_measure):
                cur = x_meas_start_idx + offset
                coord = stab_coord(x_stab_coords, s_idx)
                
                if last_x_meas[s_idx] is None:
                    add_detector(coord, [cur], t=0.0)
                else:
                    add_detector(coord, [last_x_meas[s_idx], cur], t=0.0)
                
                last_x_meas[s_idx] = cur
            
            # Measure Z-ancillas in one batch (if any)
            z_meas_start_idx = m_index
            if z_ancillas_to_measure:
                z_qubits = [a for _, a in z_ancillas_to_measure]
                c.append("MR", z_qubits)
                m_index += len(z_qubits)
            
            # Create time-like detectors for Z-ancillas
            for offset, (s_idx, _) in enumerate(z_ancillas_to_measure):
                cur = z_meas_start_idx + offset
                coord = stab_coord(z_stab_coords, s_idx)
                
                if last_z_meas[s_idx] is None:
                    add_detector(coord, [cur], t=0.0)
                else:
                    add_detector(coord, [last_z_meas[s_idx], cur], t=0.0)
                
                last_z_meas[s_idx] = cur
            
            # Update last_x_meas for all X-ancillas (they are now measured in every round)
            for offset, (s_idx, _) in enumerate(x_ancillas_to_measure):
                last_x_meas[s_idx] = x_meas_start_idx + offset

            if data_coords is not None:
                c.append("SHIFT_COORDS", [], [0.0, 0.0, 1.0])

        # ---- Final data measurement + space-like detectors ----------------
        if n:
            # ALWAYS measure ALL data qubits to ensure space-like detectors have complete information.
            # The logical operator support is used for the observable, not for measurement selection.
            qubits_to_measure = data_qubits
            
            # Measure in chosen basis. For X-basis, rotate with H first.
            if basis == "X":
                c.append("H", qubits_to_measure)
            # Remove disentangling CX gates: do NOT entangle data with ancillas before measurement.
            # Measure data qubits deterministically.
            #
            # REPLACE demolition measurement (MR) with standard M for data qubits.
            c.append("M", qubits_to_measure)

            first_data_idx = m_index
            # Track which qubits were actually measured
            measured_qubits = list(qubits_to_measure)
            data_meas_indices_all = {}  # qubit_index -> measurement_record_index
            for i, q in enumerate(measured_qubits):
                data_meas_indices_all[q] = first_data_idx + i
            m_index += len(measured_qubits)

            # Choose which stabiliser layer to use for *space-like* detectors.
            # We pair the final data measurements with the last round of the
            # same-type stabiliser checks:
            #   - Z-basis memory (|0_L>): use Z checks (hz) and their ancillas.
            #   - X-basis memory (|+_L>): use X checks (hx) and their ancillas.
            if basis == "Z":
                stab_mat = hz
                last_stab_meas = last_z_meas
                stab_coords = z_stab_coords
            else:  # basis == "X"
                stab_mat = hx
                last_stab_meas = last_x_meas
                stab_coords = x_stab_coords

            # Create space-like detectors combining final data measurements with last stabilizer measurements.
            # These detectors represent the parity checks: each stabilizer measures a product of data qubits.
            # We do this regardless of whether the observable covers all stabilizer qubits, since Stim's
            # native implementation also generates these detectors.
            obs_subset = set(measured_qubits) if measured_qubits else set(range(n))
            stab_all_qubits = set()
            if stab_mat is not None and stab_mat.size > 0:
                for row in stab_mat:
                    stab_all_qubits.update(np.where(row == 1)[0])
            
            should_create_space_like = stab_mat is not None and stab_mat.size > 0
            
            if should_create_space_like and stab_mat is not None and stab_mat.size > 0:
                num_stab = stab_mat.shape[0]
                for s_idx in range(num_stab):
                    row = stab_mat[s_idx]
                    # Data measurement indices that participate in this stabiliser.
                    data_idxs = [
                        data_meas_indices_all[q]
                        for q in np.where(row == 1)[0]
                        if q in data_meas_indices_all
                    ]
                    recs = list(data_idxs)
                    # For Z-basis: pair with last Z-ancilla measurements.
                    # For X-basis: pair with last X-ancilla measurements.
                    if basis == "Z":
                        # Z-basis: use Z-ancilla meas (last_z_meas)
                        if s_idx < len(last_z_meas) and last_z_meas[s_idx] is not None:
                            recs.append(last_z_meas[s_idx])
                    else:
                        # X-basis: use X-ancilla meas (last_x_meas)
                        if s_idx < len(last_x_meas) and last_x_meas[s_idx] is not None:
                            recs.append(last_x_meas[s_idx])
                    if not recs:
                        continue
                    # Use correct spacetime coordinates for detector
                    # Note: Stim uses t=1.0 for space-like detectors (final measurement layer)
                    coord = stab_coord(stab_coords, s_idx)
                    add_detector(coord, recs, t=1.0)

            # ---- Logical observable ---------------------------------------
            rec_indices_by_data = data_meas_indices_all

            # Derive logical support directly from logical operators (preferred)
            # This removes redundant metadata dependency
            def pauli_at(pauli_obj, q: int) -> str:
                if isinstance(pauli_obj, str):
                    return pauli_obj[q]
                return pauli_obj[q]

            logical_support: list[int] = []
            if basis == "Z" and code.logical_z_ops and self.logical_qubit < len(code.logical_z_ops):
                L = code.logical_z_ops[self.logical_qubit]
                logical_support = [q for q in range(n) if pauli_at(L, q) in ("Z", "Y")]
            elif basis == "X" and code.logical_x_ops and self.logical_qubit < len(code.logical_x_ops):
                L = code.logical_x_ops[self.logical_qubit]
                logical_support = [q for q in range(n) if pauli_at(L, q) in ("X", "Y")]

            if not logical_support:
                logical_support = measured_qubits

            obs_rec_indices = [
                rec_indices_by_data[q] for q in logical_support if q in rec_indices_by_data
            ]
            if not obs_rec_indices:
                obs_rec_indices = list(data_meas_indices_all.values())

            # Convert absolute indices -> lookbacks for OBSERVABLE_INCLUDE.
            lookbacks = [idx - m_index for idx in obs_rec_indices]
            obs_targets = [stim.target_rec(lb) for lb in lookbacks]
            c.append("OBSERVABLE_INCLUDE", obs_targets, 0)

        # NOTE: Final measurement block on data qubits uses M (not MR).
        #       Detector coordinates above use correct spacetime (x, y, t).
        #       All DETECTOR rec indices reference valid measurement results.

        return c
