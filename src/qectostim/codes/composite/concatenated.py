# src/qectostim/codes/composite/concatenated.py
from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple

import numpy as np

from qectostim.codes.abstract_code import Code, PauliString
from qectostim.codes.abstract_css import TopologicalCSSCode, CSSCode

Coord2D = Tuple[float, float]
class ConcatenatedCode(Code):
    """
    Abstract base class for concatenated codes: outer ∘ inner.

    It just remembers the outer and inner code objects and basic sizing;
    CSS-specific stuff lives in ConcatenatedCSSCode.
    """

    def __init__(self, outer: Code, inner: Code) -> None:
        # Don't call super().__init__ here; CSSCode / TopologicalCSSCode
        # will take care of base init.
        self.outer = outer
        self.inner = inner
        self._n_outer = outer.n
        self._n_inner = inner.n

    @property
    def n_outer(self) -> int:
        return self._n_outer

    @property
    def n_inner(self) -> int:
        return self._n_inner

    @property
    def name(self) -> str:
        outer_name = getattr(self.outer, "name", type(self.outer).__name__)
        inner_name = getattr(self.inner, "name", type(self.inner).__name__)
        return f"Concatenated({outer_name}, {inner_name})"

    def outer_block_indices(self, outer_q: int) -> List[int]:
        """Return physical indices of the inner block encoding outer_q."""
        start = outer_q * self._n_inner
        return list(range(start, start + self._n_inner))
    

class ConcatenatedCSSCode(CSSCode, ConcatenatedCode):
    """
    Concatenated CSS code: outer ∘ inner.

    Each physical qubit of `outer` is encoded using `inner`.
    Both outer and inner must be CSS codes.
    """

    def __init__(
        self,
        outer: CSSCode,
        inner: CSSCode,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        # First remember outer/inner and sizes.
        ConcatenatedCode.__init__(self, outer, inner)

        n_concat = self._n_outer * self._n_inner

        # --- 1. Inner checks, block-diagonal on each outer physical qubit ----
        hx_blocks: List[np.ndarray] = []
        hz_blocks: List[np.ndarray] = []

        for outer_q in range(self._n_outer):
            offset = outer_q * self._n_inner

            # X-type inner checks
            for row in inner.hx:
                new_row = np.zeros(n_concat, dtype=np.uint8)
                for i, bit in enumerate(row):
                    if bit:
                        new_row[offset + i] ^= 1
                hx_blocks.append(new_row)

            # Z-type inner checks
            for row in inner.hz:
                new_row = np.zeros(n_concat, dtype=np.uint8)
                for i, bit in enumerate(row):
                    if bit:
                        new_row[offset + i] ^= 1
                hz_blocks.append(new_row)

        # --- 2. Outer checks lifted via inner logicals (TODO) ----------------
        # Eventually:
        #   - For each outer X-check row, replace each "1" by an inner logical X
        #     on that block and add the resulting row.
        #   - Same thing for Z-checks with inner logical Z.
        # For now we only include inner checks (block-diagonal), which makes the
        # concatenated code just n_outer decoupled copies of the inner code.
        hx = np.vstack(hx_blocks) if hx_blocks else np.zeros((0, n_concat), dtype=np.uint8)
        hz = np.vstack(hz_blocks) if hz_blocks else np.zeros((0, n_concat), dtype=np.uint8)

        # TODO: implement proper logicals as composition of outer + inner.
        logical_x: List[PauliString] = []
        logical_z: List[PauliString] = []

        meta: Dict[str, Any] = dict(metadata or {})
        meta.setdefault("outer_name", outer.name)
        meta.setdefault("inner_name", inner.name)
        meta.setdefault("n_outer", self._n_outer)
        meta.setdefault("n_inner", self._n_inner)

        # Let CSSCode set up n, k, stabilizers, etc.
        CSSCode.__init__(
            self,
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

    @property
    def name(self) -> str:
        # Use the name defined in ConcatenatedCode
        return ConcatenatedCode.name.fget(self)  # type: ignore

class ConcatenatedTopologicalCSSCode(ConcatenatedCSSCode, TopologicalCSSCode):
    """
    Concatenated topological CSS code.

    Adds a geometric layout by placing a scaled copy of the inner layout
    around each outer data qubit. We deliberately avoid touching the
    homological / chain-complex structure here (that’s a separate project).
    """

    def __init__(
        self,
        outer: TopologicalCSSCode,
        inner: TopologicalCSSCode,
        metadata: Optional[Dict[str, Any]] = None,
        scale: float = 0.4,
    ) -> None:
        # Build the concatenated CSS structure (Hx, Hz, etc.).
        ConcatenatedCSSCode.__init__(self, outer=outer, inner=inner, metadata=metadata)

        outer_coords = outer.qubit_coords()
        inner_coords = inner.qubit_coords()

        # Compute inner bounding box to centre inner blocks.
        inner_arr = np.array(inner_coords, dtype=float)
        min_inner = inner_arr.min(axis=0)
        max_inner = inner_arr.max(axis=0)
        center_inner = 0.5 * (min_inner + max_inner)

        concat_coords: List[Coord2D] = []
        for outer_q, (ox, oy) in enumerate(outer_coords):
            for (ix, iy) in inner_coords:
                x = ox + scale * (ix - center_inner[0])
                y = oy + scale * (iy - center_inner[1])
                concat_coords.append((float(x), float(y)))

        self._qubit_coords = concat_coords

    def qubit_coords(self) -> List[Coord2D]:
        return self._qubit_coords