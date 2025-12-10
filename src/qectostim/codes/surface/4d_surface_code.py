from __future__ import annotations
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from qectostim.codes.complexes.css_complex import FiveCSSChainComplex
from qectostim.codes.abstract_homological import HomologicalCode, TopologicalCode


def _build_4d_surface_chain_complex(L: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Port of your current construction based on repeated homological tensor
    products of the 1D repetition code.

    Returns (sigma4G, sigma3G, sigma2G, sigma1G).
    """
    # TODO: move your current numpy logic here, cleaned up and parameterized.
    # The end of your snippet gives exactly sigma4G, sigma3G, sigma2G, sigma1G.
    raise NotImplementedError


def _build_4d_embeddings(
    sigma4G: np.ndarray,
    sigma3G: np.ndarray,
    sigma2G: np.ndarray,
    sigma1G: np.ndarray,
    L_grid: int = 10,
) -> Dict[int, Any]:
    """
    Port the coordinate-layout algorithm from your script into a reusable
    function.

    - Use your coordsFree / coordsToVisit BFS to lay out X/Z stabilizers.
    - Build qubit -> coord mapping via averaging incident stabilizer coords.
    - Ensure uniqueness with your collision-resolution loop.
    - Return CellEmbedding for each grade you care about (0..4).
    """



class FourDSurfaceCode(HomologicalCode, TopologicalCode):
    """
    4D surface code constructed via homological tensor products of the 1D
    repetition code, as in "Experiments with the 4D surface code".

    Internally represented as a 5-chain CSS complex:
        C4 -> C3 -> C2 -> C1 -> C0
    with:
        - qubits on C2
        - Hx from sigma2G
        - Hz from sigma3G^T
        - meta-checks from sigma4G^T and sigma1G
    """

    def __init__(self, L_rep: int = 2, L_grid: int = 10, metadata: Optional[Dict[str, Any]] = None):
        sigma4G, sigma3G, sigma2G, sigma1G = _build_4d_surface_chain_complex(L=L_rep)

        # Initialize the homological / CSS side
        FiveCSSChainComplex.__init__(
            self,
            sigma4=sigma4G,
            sigma3=sigma3G,
            sigma2=sigma2G,
            sigma1=sigma1G,
            metadata=metadata,
        )

        # Build the geometric embedding
        embeddings = _build_4d_embeddings(
            sigma4G=sigma4G,
            sigma3G=sigma3G,
            sigma2G=sigma2G,
            sigma1G=sigma1G,
            L_grid=L_grid,
        )

        # Initialize topological side (diamond-inheritance)
        TopologicalCode.__init__(
            self,
            complex=self.chain_complex,
            embeddings=embeddings,
            dim=4,
            metadata=self._metadata,  # share metadata
        )
