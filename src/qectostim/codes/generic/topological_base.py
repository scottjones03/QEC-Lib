# src/qectostim/codes/generic/topological_base.py
"""
Base classes for Topological CSS codes with proper chain complex structure.

This module provides flexible base classes for topological codes that
can be initialized either from:
1. A chain complex (full algebraic structure)
2. Raw Hx/Hz matrices with metadata (for simpler construction)

Class Hierarchy:
    CSSCode
    ├── TopologicalCSSCode2D - 3-chain (C2 → C1 → C0), 2D geometry
    │   ├── SurfaceCodeBase - Planar surface codes
    │   ├── ToricCodeBase - Toric codes with periodic boundaries
    │   └── ColorCodeBase2D - 2D color codes
    ├── TopologicalCSSCode3D - 4-chain (C3 → C2 → C1 → C0), 3D geometry
    │   ├── ToricCode3DBase - 3D toric codes
    │   └── ColorCode3DBase - 3D color codes
    └── TopologicalCSSCode4D - 5-chain (C4 → C3 → C2 → C1 → C0), 4D geometry
        └── TesseractCodeBase - 4D toric/tesseract codes

Chain Length Mapping:
    - chain_length = 2: Repetition codes (C1 → C0)
    - chain_length = 3: 2D CSS codes (C2 → C1 → C0)
    - chain_length = 4: 3D CSS codes (C3 → C2 → C1 → C0)
    - chain_length = 5: 4D CSS codes (C4 → C3 → C2 → C1 → C0)

Dimension Mapping:
    - dimension = 1: Linear chain (repetition)
    - dimension = 2: Planar (surface, toric, color)
    - dimension = 3: 3D (3D toric, 3D color)
    - dimension = 4: 4D (tesseract, hypergraph products)
"""
from __future__ import annotations
from abc import ABC
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from qectostim.codes.abstract_css import CSSCode, TopologicalCSSCode, TopologicalCSSCode3D, TopologicalCSSCode4D
from qectostim.codes.abstract_code import PauliString, CellEmbedding

if TYPE_CHECKING:
    from qectostim.codes.complexes.css_complex import CSSChainComplex3

Coord2D = Tuple[float, float]
Coord3D = Tuple[float, float, float]
Coord = Tuple[float, ...]


class SurfaceCodeBase(TopologicalCSSCode):
    """
    Base class for planar surface codes.
    
    Surface codes are 2D topological codes with:
    - Open boundary conditions
    - Qubits on edges of a planar graph
    - X-stabilizers on vertices, Z-stabilizers on faces (or vice versa)
    - chain_length = 3, dimension = 2
    
    Subclasses: RotatedSurfaceCode, PlanarSurfaceCode, XZZXSurfaceCode
    """
    
    @classmethod
    def from_matrices(
        cls,
        hx: np.ndarray,
        hz: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        qubit_coords: Optional[List[Coord2D]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "SurfaceCodeBase":
        """
        Create a surface code from raw matrices.
        
        This is a convenience constructor for when you don't have
        a full chain complex object.
        """
        from qectostim.codes.complexes.css_complex import CSSChainComplex3
        
        n = hx.shape[1]
        
        # Create minimal chain complex
        boundary_2 = hx.T.astype(np.uint8)
        boundary_1 = hz.astype(np.uint8)
        
        chain_complex = CSSChainComplex3(boundary_2=boundary_2, boundary_1=boundary_1)
        
        # Set up embeddings
        embeddings = {}
        if qubit_coords is not None:
            embeddings[1] = CellEmbedding(grade=1, coords=qubit_coords)
        
        meta = dict(metadata or {})
        meta["code_type"] = "surface"
        
        # Create instance using parent constructor
        instance = cls.__new__(cls)
        TopologicalCSSCode.__init__(
            instance,
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            embeddings=embeddings,
            metadata=meta,
        )
        return instance


class ToricCodeBase(TopologicalCSSCode):
    """
    Base class for 2D toric codes.
    
    Toric codes are 2D topological codes with:
    - Periodic boundary conditions (torus topology)
    - Qubits on edges of a square lattice
    - X-stabilizers on vertices, Z-stabilizers on faces
    - k = 2 logical qubits (two non-contractible loops)
    - chain_length = 3, dimension = 2
    
    Subclasses: ToricCode, ToricCode33, ToricCode55
    """
    
    @classmethod
    def from_lattice(
        cls,
        Lx: int,
        Ly: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "ToricCodeBase":
        """
        Create a toric code on an Lx × Ly lattice.
        """
        # Build Lx × Ly toric code
        n_qubits = 2 * Lx * Ly  # Two edges per vertex
        
        # This would be implemented by the subclass
        raise NotImplementedError("Use ToricCode class directly")


class ColorCodeBase2D(TopologicalCSSCode):
    """
    Base class for 2D color codes.
    
    Color codes are CSS codes on 3-colorable lattices where:
    - Both X and Z stabilizers are face operators
    - Transversal Clifford gates are available
    - chain_length = 3, dimension = 2
    
    Subclasses: TriangularColourCode, HexagonalColourCode, ColourCode488
    """
    pass


class ToricCode3DBase(TopologicalCSSCode3D):
    """
    Base class for 3D toric codes.
    
    3D toric codes have:
    - Qubits on edges of a 3D lattice
    - X-stabilizers on vertices, Z-stabilizers on cubes
    - k = 3 logical qubits (three independent loops)
    - chain_length = 4, dimension = 3
    
    Subclasses: ToricCode3D, ToricCode3DFaces
    """
    pass


class ColorCode3DBase(TopologicalCSSCode3D):
    """
    Base class for 3D color codes.
    
    3D color codes have:
    - 4-colorable lattice structure
    - X and Z stabilizers on cells
    - Transversal T gate (magic state distillation)
    - chain_length = 4, dimension = 3
    
    Subclasses: ColorCode3D, ColorCode3DPrism
    """
    pass


class TesseractCodeBase(TopologicalCSSCode4D):
    """
    Base class for 4D toric/tesseract codes.
    
    4D toric codes (tesseract codes) have:
    - Qubits on faces (2-cells) of a 4D lattice
    - X-stabilizers on edges, Z-stabilizers on 3-cells
    - k = 6 logical qubits
    - chain_length = 5, dimension = 4
    - Can be constructed via homological product of two 2D toric codes
    
    Subclasses: ToricCode4D, LoopToricCode4D
    """
    pass


class FractonCodeBase(CSSCode):
    """
    Base class for fracton codes.
    
    Fracton codes are 3D topological codes with restricted mobility:
    - Isolated excitations cannot move freely
    - Type-I: lineons and planons (XCubeCode, ChamonCode)
    - Type-II: completely immobile fractons (HaahCode)
    
    Fracton codes have unusual properties:
    - Not described by standard TQFT
    - Ground state degeneracy depends on system size
    - chain_length = 3 (but with non-standard structure)
    """
    
    def __init__(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        meta = dict(metadata or {})
        meta["is_fracton"] = True
        meta.setdefault("chain_length", 3)
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

    @property
    def fracton_type(self) -> str:
        """Return 'type-I' or 'type-II' for this fracton code."""
        return self._metadata.get("fracton_type", "unknown")


class XCubeCodeBase(FractonCodeBase):
    """
    Base class for X-cube type fracton codes.
    
    Type-I fracton codes with lineon excitations:
    - Excitations can move along lines
    - XCubeCode, CheckerboardCode
    """
    
    def __init__(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        meta = dict(metadata or {})
        meta["fracton_type"] = "type-I"
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)


class HaahCodeBase(FractonCodeBase):
    """
    Base class for Haah-type fracton codes.
    
    Type-II fracton codes with completely immobile excitations:
    - No string operators
    - Immobile fracton excitations
    - HaahCode (cubic code)
    """
    
    def __init__(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        meta = dict(metadata or {})
        meta["fracton_type"] = "type-II"
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)


class RepetitionCodeBase(CSSCode):
    """
    Base class for repetition codes.
    
    Repetition codes are the simplest CSS codes with:
    - 2-chain structure (C1 → C0)
    - Only Z-stabilizers (adjacent-pair checks)
    - No X-stabilizers (only detects X errors, not Z)
    - k = 1, d = n
    - chain_length = 2, dimension = 1
    
    Note: This is NOT a TopologicalCSSCode because it's a 2-chain,
    not a 3-chain. The lack of X stabilizers means it only protects
    against X errors in one logical basis.
    """
    
    def __init__(
        self,
        hx: np.ndarray,
        hz: np.ndarray,
        logical_x: List[PauliString],
        logical_z: List[PauliString],
        metadata: Optional[Dict[str, Any]] = None,
    ):
        meta = dict(metadata or {})
        meta["chain_length"] = 2  # 2-chain: C1 → C0
        meta["code_type"] = "repetition"
        
        super().__init__(
            hx=hx,
            hz=hz,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )

    @property
    def chain_length(self) -> int:
        """Repetition codes are 2-chain complexes."""
        return 2
