# src/qectostim/decoders/concatenated_decoder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import stim

from qectostim.codes.composite.concatenated import ConcatenatedCode
from qectostim.decoders.base import Decoder
from qectostim.decoders.decoder_selector import select_decoder


@dataclass
class ConcatenatedDecoder(Decoder):
    """Hierarchical decoder for concatenated codes.

    This decoder performs level-by-level decoding, starting from the lowest
    (innermost) code and working outwards to the outermost code.

    It relies on metadata attached to the ConcatenatedCode specifying how
    the global DEM factors into per-level DEMs and detector index slices.
    """

    code: ConcatenatedCode
    dem: stim.DetectorErrorModel
    preferred: Optional[str] = None

    # Extra knobs passed down to select_decoder for each level
    physical_error_rate: float = 1e-3
    max_bp_iters: int = 50
    osd_order: int = 2
    tesseract_bond_dim: int = 16
    belief_damping: float = 0.0

    def __post_init__(self) -> None:
        meta = getattr(self.code, "metadata", getattr(self.code, "_metadata", {}))
        concat_meta = meta.get("concatenation", {})

        # Expected structure (you can adjust to your actual metadata):
        #   dem_per_level: List[stim.DetectorErrorModel]
        #   dem_slices: List[Dict[block_id, Tuple[start, stop]]]
        #   logicals_per_level: List[int]
        self.dem_per_level: List[stim.DetectorErrorModel] = concat_meta["dem_per_level"]
        self.dem_slices: List[Dict[int, Tuple[int, int]]] = concat_meta["dem_slices"]
        self.logicals_per_level: List[int] = concat_meta["logicals_per_level"]

        self.levels = len(self.dem_per_level)
        if self.levels != len(self.dem_slices):
            raise ValueError(
                "ConcatenatedDecoder: dem_per_level and dem_slices must have same length."
            )

        # Build a decoder for each level.
        self._level_decoders: List[Decoder] = []
        for lvl, lvl_dem in enumerate(self.dem_per_level):
            dec = select_decoder(
                lvl_dem,
                preferred=self.preferred,
                code=None,  # inner/outer codes not needed for DEM-only decoders
                physical_error_rate=self.physical_error_rate,
                max_bp_iters=self.max_bp_iters,
                osd_order=self.osd_order,
                tesseract_bond_dim=self.tesseract_bond_dim,
                belief_damping=self.belief_damping,
            )
            self._level_decoders.append(dec)

        self.num_detectors = self.dem.num_detectors
        self.num_observables = self.dem.num_observables

    def decode_batch(self, dets: np.ndarray) -> np.ndarray:
        """Decode a batch of global detector outcomes.

        Parameters
        ----------
        dets : np.ndarray
            Shape (shots, num_detectors).

        Returns
        -------
        np.ndarray
            Shape (shots, num_observables) -- final logical flips at the top level.
        """
        dets = np.asarray(dets, dtype=np.uint8)
        if dets.ndim == 1:
            dets = dets.reshape(1, -1)

        if dets.shape[1] != self.num_detectors:
            raise ValueError(
                f"ConcatenatedDecoder: expected dets.shape[1]={self.num_detectors}, "
                f"got {dets.shape[1]}"
            )

        shots = dets.shape[0]

        # Start from the global detector outcomes at level 0 (innermost).
        # We'll maintain a list of "syndrome batches" per level.
        # At level ℓ, a shot's syndrome is the detector outcomes for *that level*
        # (possibly derived from lower levels).
        level_syndromes: List[np.ndarray] = [dets]

        # For each level, construct per-block syndromes, decode them, then
        # aggregate to the next level's syndrome.
        for lvl in range(self.levels):
            lvl_dem = self.dem_per_level[lvl]
            lvl_dec = self._level_decoders[lvl]
            block_slices = self.dem_slices[lvl]

            # Build a syndrome array for each block, then decode them independently.
            num_blocks = len(block_slices)
            lvl_num_dets = lvl_dem.num_detectors
            lvl_num_logicals = self.logicals_per_level[lvl]

            # Sanity: slices should cover exactly lvl_num_dets indices in total
            # (assuming non-overlapping slices).
            # If there is overlap or more complex mapping, you can replace this logic
            # with a custom gather/scatter according to your metadata.
            lvl_syndrome = np.zeros((shots, lvl_num_dets), dtype=np.uint8)
            for block_id, (start, stop) in block_slices.items():
                # Extract this block's portion from the previous level's syndrome.
                # At lvl == 0, we use the *global* dets; for lvl > 0 we would
                # typically replace this with a mapping from logicals of level-1.
                lvl_syndrome[:, start:stop] = level_syndromes[lvl][:, start:stop]

            # Decode each whole level syndrome in one shot (all blocks implicitly).
            lvl_corrections = lvl_dec.decode_batch(lvl_syndrome)
            lvl_corrections = np.asarray(lvl_corrections, dtype=np.uint8)
            if lvl_corrections.ndim == 1:
                lvl_corrections = lvl_corrections.reshape(-1, lvl_num_logicals)

            # Prepare next level's syndrome from these logical corrections.
            # The simplest scheme: next-level "detectors" are just these logicals.
            if lvl < self.levels - 1:
                # Next level uses these logicals as its "syndrome bits".
                # In more advanced setups you might have a linear map from lower-
                # level logicals to higher-level stabiliser syndrome.
                next_lvl_num_dets = self.dem_per_level[lvl + 1].num_detectors

                # Here we assume `lvl_num_logicals == next_lvl_num_dets`. If that
                # isn't true you can add a mapping in your metadata.
                if lvl_num_logicals != next_lvl_num_dets:
                    raise ValueError(
                        f"ConcatenatedDecoder: at level {lvl}, logicals_per_level "
                        f"({lvl_num_logicals}) != next-level num_detectors "
                        f"({next_lvl_num_dets}). Add a mapping in `concatenation` "
                        "metadata and adapt this code."
                    )

                next_lvl_syndrome = lvl_corrections.copy()
                level_syndromes.append(next_lvl_syndrome)

        # After the final level, lvl_corrections encodes the top-level logical flips.
        # For a single logical qubit, this is shape (shots, 1); for more, it is (shots, k).
        top_logicals = lvl_corrections  # from last iteration
        # Map into the DEM's observables space if needed. For now we assume they align.
        if top_logicals.shape[1] != self.num_observables:
            # If you have a mapping from top-level logical indices to DEM observables,
            # apply it here using metadata.
            raise ValueError(
                "ConcatenatedDecoder: top-level logicals dimension "
                f"{top_logicals.shape[1]} != dem.num_observables={self.num_observables}. "
                "Add an observable mapping in `concatenation` metadata and adapt."
            )

        return top_logicals