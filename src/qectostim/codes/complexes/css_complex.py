# src/qectostim/codes/complexes/css_complex.py
"""
CSS chain complex classes for different chain lengths.

Chain Complex Structure:
    - CSSChainComplex2: 2-chain (C1 → C0) - Repetition codes
    - CSSChainComplex3: 3-chain (C2 → C1 → C0) - 2D surface/toric codes  
    - CSSChainComplex4: 4-chain (C3 → C2 → C1 → C0) - 3D toric codes
    - FiveCSSChainComplex: 5-chain (C4 → C3 → C2 → C1 → C0) - 4D tesseract

The convention is:
    - boundary_k (sigma_k): C_k → C_{k-1}, shape (#C_{k-1}, #C_k)
    - Hx comes from the boundary map above qubit_grade
    - Hz comes from the transpose of the boundary map at qubit_grade
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .chain_complex import ChainComplex


@dataclass
class CSSChainComplex2(ChainComplex):
    """
    2-term chain complex for simple CSS codes like repetition code:

        C1 --∂1--> C0

    - C1: Qubits (n physical bits)
    - C0: Checks (parity checks)
    
    For the repetition code:
    - C1 has n elements (data bits)
    - C0 has n-1 elements (parity checks between adjacent bits)
    """

    boundary_1: np.ndarray  # shape: (#C0, #C1)

    def __init__(self, boundary_1: np.ndarray):
        self.boundary_1 = boundary_1
        
        boundary_maps = {
            1: boundary_1,
        }
        # Qubits live on C1 → grade 1
        ChainComplex.__init__(self, boundary_maps=boundary_maps, qubit_grade=1)

    @property
    def n_qubits(self) -> int:
        """Number of physical qubits (C1)."""
        return self.boundary_1.shape[1]

    @property
    def n_checks(self) -> int:
        """Number of checks (C0)."""
        return self.boundary_1.shape[0]

    @property
    def hz(self) -> np.ndarray:
        """Z-stabilizer parity-check matrix."""
        return (self.boundary_1.astype(np.uint8) % 2)

    @property
    def hx(self) -> np.ndarray:
        """X-stabilizer parity-check matrix (empty for 2-chain)."""
        return np.zeros((0, self.n_qubits), dtype=np.uint8)


@dataclass
class CSSChainComplex3(ChainComplex):
    """
    3-term chain complex for 2D CSS codes:

        C2 --∂2--> C1 --∂1--> C0

    Standard surface/toric code structure:
      - C2: faces (plaquettes) - X stabilizers
      - C1: edges (data qubits)
      - C0: vertices - Z stabilizers

    Parity-checks:
      - H_X := ∂2^T  (faces → edges)
      - H_Z := ∂1    (edges → vertices)
    """

    boundary_2: np.ndarray  # shape: (#C1, #C2)
    boundary_1: np.ndarray  # shape: (#C0, #C1)

    def __init__(self, boundary_2: np.ndarray, boundary_1: np.ndarray):
        self.boundary_2 = boundary_2
        self.boundary_1 = boundary_1

        boundary_maps = {
            2: boundary_2,
            1: boundary_1,
        }
        # Qubits live on C1 (edges) → grade 1
        ChainComplex.__init__(self, boundary_maps=boundary_maps, qubit_grade=1)

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

    @property
    def hx(self) -> np.ndarray:
        """X-stabilizer parity-check matrix H_X (mod 2)."""
        # ∂2: shape (#C1, #C2); H_X := ∂2^T → shape (#C2, #C1)
        return (self.boundary_2.T.astype(np.uint8) % 2)

    @property
    def hz(self) -> np.ndarray:
        """Z-stabilizer parity-check matrix H_Z (mod 2)."""
        # ∂1: shape (#C0, #C1)
        return (self.boundary_1.astype(np.uint8) % 2)


@dataclass  
class CSSChainComplex4(ChainComplex):
    """
    4-term chain complex for 3D CSS codes:

        C3 --∂3--> C2 --∂2--> C1 --∂1--> C0

    Standard 3D toric code structure:
      - C3: 3-cells (cubes)
      - C2: 2-cells (faces)
      - C1: 1-cells (edges, data qubits)
      - C0: 0-cells (vertices)

    For 3D toric with qubits on edges (grade 1):
      - H_X := ∂2^T (faces → edges)
      - H_Z := ∂1   (edges → vertices)
      - Meta-X checks from ∂3
      - Meta-Z checks from ∂1's structure
    """

    boundary_3: np.ndarray  # shape: (#C2, #C3)
    boundary_2: np.ndarray  # shape: (#C1, #C2)
    boundary_1: np.ndarray  # shape: (#C0, #C1)

    def __init__(
        self,
        boundary_3: np.ndarray,
        boundary_2: np.ndarray,
        boundary_1: np.ndarray,
        qubit_grade: int = 1,
    ):
        self.boundary_3 = boundary_3
        self.boundary_2 = boundary_2
        self.boundary_1 = boundary_1

        boundary_maps = {
            3: boundary_3,
            2: boundary_2,
            1: boundary_1,
        }
        ChainComplex.__init__(self, boundary_maps=boundary_maps, qubit_grade=qubit_grade)

    @property
    def n_cubes(self) -> int:
        """Number of 3-cells (C3)."""
        return self.boundary_3.shape[1]

    @property
    def n_faces(self) -> int:
        """Number of 2-cells (C2)."""
        return self.boundary_2.shape[1]

    @property
    def n_edges(self) -> int:
        """Number of 1-cells / edges (C1)."""
        return self.boundary_1.shape[1]

    @property
    def n_vertices(self) -> int:
        """Number of 0-cells / vertices (C0)."""
        return self.boundary_1.shape[0]

    @property
    def hx(self) -> np.ndarray:
        """X-stabilizer parity-check matrix."""
        if self.qubit_grade == 1:
            # Qubits on edges: Hx from ∂2^T
            return (self.boundary_2.T.astype(np.uint8) % 2)
        elif self.qubit_grade == 2:
            # Qubits on faces: Hx from ∂3^T
            return (self.boundary_3.T.astype(np.uint8) % 2)
        else:
            raise ValueError(f"Unsupported qubit_grade {self.qubit_grade}")

    @property
    def hz(self) -> np.ndarray:
        """Z-stabilizer parity-check matrix."""
        if self.qubit_grade == 1:
            # Qubits on edges: Hz from ∂1
            return (self.boundary_1.astype(np.uint8) % 2)
        elif self.qubit_grade == 2:
            # Qubits on faces: Hz from ∂2^T
            return (self.boundary_2.T.astype(np.uint8) % 2)
        else:
            raise ValueError(f"Unsupported qubit_grade {self.qubit_grade}")


class FiveCSSChainComplex(ChainComplex):
    """
    5-term chain complex for 4D CSS codes:

        C4 --σ4--> C3 --σ3--> C2 --σ2--> C1 --σ1--> C0

    Standard 4D toric code (tesseract) structure:
      - C4: 4-cells (hypercubes)
      - C3: 3-cells (cubes)
      - C2: 2-cells (faces, typically qubits)
      - C1: 1-cells (edges)
      - C0: 0-cells (vertices)

    With qubits on C2 (faces):
      - H_X := σ2^T (C2 → C1)
      - H_Z := σ3^T (C3 → C2)
      - Meta-X checks from σ1
      - Meta-Z checks from σ4^T
    """

    def __init__(
        self,
        sigma4: np.ndarray,
        sigma3: np.ndarray,
        sigma2: np.ndarray,
        sigma1: np.ndarray,
        qubit_grade: int = 2,
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
        super().__init__(boundary_maps=boundary_maps, qubit_grade=qubit_grade)

    @property
    def n_hypercubes(self) -> int:
        """Number of 4-cells (C4)."""
        return self.sigma4.shape[1]

    @property
    def n_cubes(self) -> int:
        """Number of 3-cells (C3)."""
        return self.sigma3.shape[1]

    @property
    def n_faces(self) -> int:
        """Number of 2-cells (C2)."""
        return self.sigma2.shape[1]

    @property
    def n_edges(self) -> int:
        """Number of 1-cells (C1)."""
        return self.sigma1.shape[1]

    @property
    def n_vertices(self) -> int:
        """Number of 0-cells (C0)."""
        return self.sigma1.shape[0]

    @property
    def hx(self) -> np.ndarray:
        """X-type stabilizers from σ3^T.
        
        For qubits on grade 2 (faces), X-stabilizers come from 
        3-cells (cubes) checking their boundary 2-cells.
        """
        return (self.sigma3.T.astype(np.uint8) % 2)

    @property
    def hz(self) -> np.ndarray:
        """Z-type stabilizers from σ2.
        
        For qubits on grade 2 (faces), Z-stabilizers come from
        1-cells (edges) checking their incident 2-cells.
        """
        return (self.sigma2.astype(np.uint8) % 2)

    @property
    def meta_x(self) -> np.ndarray:
        """X-type meta-checks from σ1."""
        return (self.sigma1.astype(np.uint8) % 2)

    @property
    def meta_z(self) -> np.ndarray:
        """Z-type meta-checks from σ4^T."""
        return (self.sigma4.T.astype(np.uint8) % 2)
