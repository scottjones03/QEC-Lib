from dataclasses import dataclass
import numpy as np

from .chain_complex import ChainComplex


@dataclass
class CSSChainComplex3(ChainComplex):
    """
    3-term chain complex for a CSS code:

        C2 --∂2--> C1 --∂1--> C0

    In a surface/colour-code picture:
      - C2 ~ faces (plaquettes)
      - C1 ~ edges (data qubits)
      - C0 ~ vertices

    Canonical CSS parity-checks:
      - H_X := ∂2^T  (faces → edges)
      - H_Z := ∂1    (edges → vertices)
    """

    boundary_2: np.ndarray  # shape: (#C1, #C2)
    boundary_1: np.ndarray  # shape: (#C0, #C1)

    def __init__(self, boundary_2: np.ndarray, boundary_1: np.ndarray):
        # Store raw boundaries
        self.boundary_2 = boundary_2
        self.boundary_1 = boundary_1

        # Build underlying ChainComplex structure.
        # Grades: 2, 1, 0 with:
        #   sigma_2 = ∂2, sigma_1 = ∂1
        boundary_maps = {
            2: boundary_2,
            1: boundary_1,
        }
        # Qubits live on C1 (edges) → grade 1.
        ChainComplex.__init__(self, boundary_maps=boundary_maps, qubit_grade=1)

    # --- basic dimensions ---

    @property
    def n_edges(self) -> int:
        """Number of data qubits (edges = C1)."""
        return self.boundary_1.shape[1]

    @property
    def n_faces(self) -> int:
        """Number of 2-cells / faces (= C2)."""
        return self.boundary_2.shape[1]

    @property
    def n_vertices(self) -> int:
        """Number of 0-cells / vertices (= C0)."""
        return self.boundary_1.shape[0]

    # --- CSS parity-check matrices ---

    @property
    def hx(self) -> np.ndarray:
        """X-stabilizer parity-check matrix H_X (mod 2)."""
        # ∂2: shape (#C1, #C2); H_X := ∂2^T → shape (#C2, #C1)
        return (self.boundary_2.T.astype(np.uint8) % 2)

    @property
    def hz(self) -> np.ndarray:
        """Z-stabilizer parity-check matrix H_Z (mod 2)."""
        # ∂1: shape (#C0, #C1); treat rows as Z checks acting on edges (C1).
        return (self.boundary_1.astype(np.uint8) % 2)


class FiveCSSChainComplex(ChainComplex):
    """
    5-step CSS chain complex:

        C4 --sigma4--> C3 --sigma3--> C2 --sigma2--> C1 --sigma1--> C0

    with:
        - data qubits on C2
        - X checks from sigma2 (generalising ∂2)
        - Z checks from sigma3^T (generalising ∂3^T)
        - X meta-checks from sigma1
        - Z meta-checks from sigma4^T

    Shapes:
        sigma_k has shape (#C_{k-1}, #C_k).

    So:
        - sigma2: (#C1, #C2)
        - sigma3: (#C2, #C3)
        - sigma1: (#C0, #C1)
        - sigma4: (#C3, #C4)
    """

    def __init__(
        self,
        sigma4: np.ndarray,
        sigma3: np.ndarray,
        sigma2: np.ndarray,
        sigma1: np.ndarray,
    ):
        self.sigma4 = sigma4
        self.sigma3 = sigma3
        self.sigma2 = sigma2
        self.sigma1 = sigma1

        boundary_maps = {
            4: sigma4,
            3: sigma3,
            2: sigma2,
            1: sigma1,
        }
        # Qubits on C2 → grade 2.
        super().__init__(boundary_maps=boundary_maps, qubit_grade=2)

    # --- CSS parity checks and meta-checks ---

    @property
    def hx(self) -> np.ndarray:
        """
        X-type stabilizers H_X.

        X checks come from sigma2 (C2→C1), generalising H_X := ∂2^T in 2D:
            sigma2: (#C1, #C2)
            H_X := sigma2^T: (#C2, #C1)
        """
        return (self.sigma2.T.astype(np.uint8) % 2)

    @property
    def hz(self) -> np.ndarray:
        """
        Z-type stabilizers H_Z.

        Z checks come from sigma3^T:
            sigma3: (#C2, #C3)
            H_Z := sigma3^T: (#C3, #C2)
        """
        return (self.sigma3.T.astype(np.uint8) % 2)

    @property
    def meta_x(self) -> np.ndarray:
        """
        X-type meta-checks (syndrome-of-syndrome in the X sector).

        Taken directly from sigma1 (C1→C0):
            sigma1: (#C0, #C1)
        """
        return (self.sigma1.astype(np.uint8) % 2)

    @property
    def meta_z(self) -> np.ndarray:
        """
        Z-type meta-checks (syndrome-of-syndrome in the Z sector).

        Taken from sigma4^T:
            sigma4: (#C3, #C4)
            meta_Z := sigma4^T: (#C4, #C3)
        """
        return (self.sigma4.T.astype(np.uint8) % 2)