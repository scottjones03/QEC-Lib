# src/qectostim/experiments/stabilizer_rounds/color.py
"""
Color code stabilizer round builder with Chromobius-compatible detectors.

This module provides ColorCodeStabilizerRoundBuilder which extends
CSSStabilizerRoundBuilder to emit 4D detector coordinates encoding
(basis, color) as required by the Chromobius decoder.
"""
from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

from .context import DetectorContext
from .css import CSSStabilizerRoundBuilder

if TYPE_CHECKING:
    from qectostim.codes.abstract_css import CSSCode


class ColorCodeStabilizerRoundBuilder(CSSStabilizerRoundBuilder):
    """
    Stabilizer round builder for color codes with Chromobius-compatible detectors.
    
    Extends CSSStabilizerRoundBuilder to emit detector coordinates with a 4th
    component encoding (basis, color) as required by the Chromobius decoder:
    
    - coord[3] = 0, 1, 2 for X-type red, green, blue stabilizers
    - coord[3] = 3, 4, 5 for Z-type red, green, blue stabilizers
    
    The code must provide:
    - metadata["stab_colors"]: list of colors (0=red, 1=green, 2=blue) per stabilizer
    - metadata["is_chromobius_compatible"]: True
    """
    
    def __init__(
        self,
        code: "CSSCode",
        ctx: DetectorContext,
        block_name: str = "main",
        data_offset: int = 0,
        ancilla_offset: Optional[int] = None,
        measurement_basis: str = "Z",
    ):
        super().__init__(code, ctx, block_name, data_offset, ancilla_offset, measurement_basis)
        
        # Validate and cache color metadata
        if not self._meta.get("is_chromobius_compatible", False):
            raise ValueError(
                "ColorCodeStabilizerRoundBuilder requires a code with "
                "metadata['is_chromobius_compatible'] = True"
            )
        if "stab_colors" not in self._meta:
            raise ValueError(
                "ColorCodeStabilizerRoundBuilder requires code.metadata['stab_colors'] "
                "to be a list of colors (0=red, 1=green, 2=blue) per stabilizer"
            )
        
        self._stab_colors = self._meta["stab_colors"]
    
    def _get_color(self, s_idx: int, is_x_type: bool) -> int:
        """
        Get Chromobius color encoding for stabilizer.
        
        For X-type: color in {0, 1, 2}
        For Z-type: color + 3 in {3, 4, 5}
        """
        base_color = self._stab_colors[s_idx % len(self._stab_colors)] if self._stab_colors else 0
        return base_color if is_x_type else base_color + 3
    
    def _get_stab_coord(self, stab_type: str, s_idx: int) -> Tuple[float, float, float, float]:
        """Get 4D detector coordinate for a stabilizer (x, y, t, color)."""
        is_x_type = (stab_type == "x")
        coords = self._x_stab_coords if is_x_type else self._z_stab_coords
        color = float(self._get_color(s_idx, is_x_type))
        
        if s_idx < len(coords):
            x, y = coords[s_idx][:2]
            return (float(x), float(y), self.ctx.current_time, color)
        return (0.0, 0.0, self.ctx.current_time, color)
