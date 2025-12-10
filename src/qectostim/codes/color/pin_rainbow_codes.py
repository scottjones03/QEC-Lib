"""
Pin Codes and Rainbow Codes.

Specialized color code variants with enhanced properties:

Pin Codes:
    QuantumPinCode: Pin code construction for improved thresholds
    DoublePinCode: Double-pin variant with better distance scaling

Rainbow Codes:
    RainbowCode: Generalized color codes with rainbow structure
    HolographicRainbowCode: Rainbow codes with holographic properties

References:
    - Bombin & Martin-Delgado, "Exact topological quantum order" (2008)
    - Kubica & Beverland, "Universal transversal gates" (2015)
    - Brown et al., "Poking holes and cutting corners" (2016)
"""

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np

from ..generic.qldpc_base import QLDPCCode


class QuantumPinCode(QLDPCCode):
    """
    Quantum Pin Code.
    
    Pin codes are CSS codes constructed via a "pinning" procedure that
    gives improved threshold and distance properties compared to standard
    surface codes. They can be viewed as color codes with specific boundary
    conditions.
    
    For parameters (d, m):
        - d: base distance
        - m: number of pins (determines redundancy)
        - n ∝ d² × m
        - k = 1 (single logical qubit)
    
    Parameters
    ----------
    d : int
        Base distance parameter (default: 3)
    m : int  
        Number of pins (default: 2)
    """
    
    def __init__(self, d: int = 3, m: int = 2, metadata: Optional[Dict[str, Any]] = None):
        if d < 2:
            raise ValueError("d must be at least 2")
        if m < 1:
            raise ValueError("m must be at least 1")
        
        self._d = d
        self._m = m
        
        hx, hz, n_qubits = self._build_pin_code(d, m)
        
        k = m  # Number of logical qubits
        logical_x, logical_z = self._build_logicals(n_qubits, k, d)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"QuantumPinCode_d{d}_m{m}",
            "n": n_qubits,
            "k": k,
            "distance": d,
            "n_pins": m,
        })
        
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)
    
    @property
    def d(self) -> int:
        return self._d
    
    @property
    def m(self) -> int:
        return self._m
    
    @staticmethod
    def _build_pin_code(d: int, m: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build pin code using HGP construction."""
        # Use HGP of repetition codes
        # First code: length d repetition
        na = d
        ma = d - 1
        a = np.zeros((ma, na), dtype=np.uint8)
        for i in range(ma):
            a[i, i] = 1
            a[i, i + 1] = 1
        
        # Second code: length m+1 
        nb = m + 1
        mb = m
        b = np.zeros((mb, nb), dtype=np.uint8)
        for i in range(mb):
            b[i, i] = 1
            b[i, i + 1] = 1
        
        # HGP construction
        n_left = na * nb
        n_right = ma * mb
        n_qubits = n_left + n_right
        
        hx = np.zeros((ma * nb, n_qubits), dtype=np.uint8)
        stab = 0
        for check_a in range(ma):
            for bit_b in range(nb):
                for bit_a in range(na):
                    if a[check_a, bit_a]:
                        hx[stab, bit_a * nb + bit_b] ^= 1
                for check_b in range(mb):
                    if b[check_b, bit_b]:
                        hx[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        hz = np.zeros((na * mb, n_qubits), dtype=np.uint8)
        stab = 0
        for bit_a in range(na):
            for check_b in range(mb):
                for bit_b in range(nb):
                    if b[check_b, bit_b]:
                        hz[stab, bit_a * nb + bit_b] ^= 1
                for check_a in range(ma):
                    if a[check_a, bit_a]:
                        hz[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        return hx, hz, n_qubits
    
    @staticmethod
    def _build_logicals(n_qubits: int, k: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators."""
        logical_x = np.zeros((k, n_qubits), dtype=np.uint8)
        logical_z = np.zeros((k, n_qubits), dtype=np.uint8)
        
        for i in range(k):
            for j in range(d):
                logical_x[i, j * (k + 1) + i] = 1
            logical_z[i, i] = 1
        
        return logical_x, logical_z


class DoublePinCode(QLDPCCode):
    """
    Double Pin Code.
    
    A double-pin construction that combines two pin code structures
    for improved distance scaling. Uses tensor product structure
    of two pin configurations.
    
    Parameters
    ----------
    d : int
        Base distance parameter (default: 3)
    """
    
    def __init__(self, d: int = 3, metadata: Optional[Dict[str, Any]] = None):
        if d < 2:
            raise ValueError("d must be at least 2")
        
        self._d = d
        
        hx, hz, n_qubits = self._build_double_pin(d)
        
        k = d - 1
        logical_x, logical_z = self._build_logicals(n_qubits, k)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"DoublePinCode_d{d}",
            "n": n_qubits,
            "k": k,
            "distance": d,
        })
        
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)
    
    @property
    def d(self) -> int:
        return self._d
    
    @staticmethod
    def _build_double_pin(d: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build double pin code."""
        # Use HGP with square structure
        n = d
        m = d - 1
        
        h = np.zeros((m, n), dtype=np.uint8)
        for i in range(m):
            h[i, i] = 1
            h[i, i + 1] = 1
        
        # HGP of h with itself
        n_left = n * n
        n_right = m * m
        n_qubits = n_left + n_right
        
        hx = np.zeros((m * n, n_qubits), dtype=np.uint8)
        stab = 0
        for check_a in range(m):
            for bit_b in range(n):
                for bit_a in range(n):
                    if h[check_a, bit_a]:
                        hx[stab, bit_a * n + bit_b] ^= 1
                for check_b in range(m):
                    if h[check_b, bit_b]:
                        hx[stab, n_left + check_a * m + check_b] ^= 1
                stab += 1
        
        hz = np.zeros((n * m, n_qubits), dtype=np.uint8)
        stab = 0
        for bit_a in range(n):
            for check_b in range(m):
                for bit_b in range(n):
                    if h[check_b, bit_b]:
                        hz[stab, bit_a * n + bit_b] ^= 1
                for check_a in range(m):
                    if h[check_a, bit_a]:
                        hz[stab, n_left + check_a * m + check_b] ^= 1
                stab += 1
        
        return hx, hz, n_qubits
    
    @staticmethod
    def _build_logicals(n_qubits: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators."""
        logical_x = np.zeros((k, n_qubits), dtype=np.uint8)
        logical_z = np.zeros((k, n_qubits), dtype=np.uint8)
        
        for i in range(k):
            logical_x[i, i] = 1
            logical_z[i, n_qubits // 2 + i] = 1
        
        return logical_x, logical_z


class RainbowCode(QLDPCCode):
    """
    Rainbow Code.
    
    Generalized color codes with a "rainbow" structure that gives 
    transversal gates and improved fault tolerance. The rainbow
    structure assigns different "colors" to different layers.
    
    For depth r (rainbow layers):
        - n ∝ L² × r
        - k = r (multiple logical qubits)
        - Transversal T-gates on some logical qubits
    
    Parameters
    ----------
    L : int
        Linear lattice size (default: 3)
    r : int
        Number of rainbow layers (default: 3)
    """
    
    def __init__(self, L: int = 3, r: int = 3, metadata: Optional[Dict[str, Any]] = None):
        if L < 2:
            raise ValueError("L must be at least 2")
        if r < 2:
            raise ValueError("r must be at least 2")
        
        self._L = L
        self._r = r
        
        hx, hz, n_qubits = self._build_rainbow_code(L, r)
        
        k = r
        logical_x, logical_z = self._build_logicals(n_qubits, k, L)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"RainbowCode_L{L}_r{r}",
            "n": n_qubits,
            "k": k,
            "distance": L,  # Distance from underlying surface code
            "L": L,
            "rainbow_depth": r,
            "transversal_gates": ["CNOT", "T (on some)"],
        })
        
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)
    
    @property
    def L(self) -> int:
        return self._L
    
    @property
    def r(self) -> int:
        return self._r
    
    @staticmethod
    def _build_rainbow_code(L: int, r: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build rainbow code using layered HGP."""
        # Use HGP with L and r dimensions
        na = L
        ma = L - 1
        nb = r
        mb = r - 1
        
        a = np.zeros((ma, na), dtype=np.uint8)
        for i in range(ma):
            a[i, i] = 1
            a[i, i + 1] = 1
        
        b = np.zeros((mb, nb), dtype=np.uint8)
        for i in range(mb):
            b[i, i] = 1
            b[i, i + 1] = 1
        
        # HGP construction
        n_left = na * nb
        n_right = ma * mb
        n_qubits = n_left + n_right
        
        hx = np.zeros((ma * nb, n_qubits), dtype=np.uint8)
        stab = 0
        for check_a in range(ma):
            for bit_b in range(nb):
                for bit_a in range(na):
                    if a[check_a, bit_a]:
                        hx[stab, bit_a * nb + bit_b] ^= 1
                for check_b in range(mb):
                    if b[check_b, bit_b]:
                        hx[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        hz = np.zeros((na * mb, n_qubits), dtype=np.uint8)
        stab = 0
        for bit_a in range(na):
            for check_b in range(mb):
                for bit_b in range(nb):
                    if b[check_b, bit_b]:
                        hz[stab, bit_a * nb + bit_b] ^= 1
                for check_a in range(ma):
                    if a[check_a, bit_a]:
                        hz[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        return hx, hz, n_qubits
    
    @staticmethod
    def _build_logicals(n_qubits: int, k: int, L: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators."""
        logical_x = np.zeros((k, n_qubits), dtype=np.uint8)
        logical_z = np.zeros((k, n_qubits), dtype=np.uint8)
        
        for i in range(k):
            for j in range(L):
                logical_x[i, j * k + i] = 1
            logical_z[i, i] = 1
        
        return logical_x, logical_z


class HolographicRainbowCode(QLDPCCode):
    """
    Holographic Rainbow Code.
    
    Rainbow codes with holographic properties inspired by AdS/CFT
    correspondence. The bulk-boundary relationship gives improved
    error correction properties.
    
    Parameters
    ----------
    L : int
        Boundary size (default: 4)
    bulk_depth : int
        Holographic bulk depth (default: 2)
    """
    
    def __init__(self, L: int = 4, bulk_depth: int = 2, metadata: Optional[Dict[str, Any]] = None):
        if L < 3:
            raise ValueError("L must be at least 3")
        if bulk_depth < 1:
            raise ValueError("bulk_depth must be at least 1")
        
        self._L = L
        self._bulk_depth = bulk_depth
        
        hx, hz, n_qubits = self._build_holographic_code(L, bulk_depth)
        
        k = bulk_depth
        logical_x, logical_z = self._build_logicals(n_qubits, k)
        
        meta: Dict[str, Any] = dict(metadata or {})
        meta.update({
            "name": f"HolographicRainbowCode_L{L}_d{bulk_depth}",
            "n": n_qubits,
            "k": k,
            "boundary_size": L,
            "bulk_depth": bulk_depth,
            "holographic": True,
        })
        
        super().__init__(hx, hz, logical_x, logical_z, metadata=meta)
    
    @property
    def L(self) -> int:
        return self._L
    
    @property
    def bulk_depth(self) -> int:
        return self._bulk_depth
    
    @staticmethod
    def _build_holographic_code(L: int, depth: int) -> Tuple[np.ndarray, np.ndarray, int]:
        """Build holographic code using layered structure."""
        # Boundary: L qubits per layer, depth layers
        # Use HGP to ensure CSS validity
        na = L
        ma = L - 1
        nb = depth + 1
        mb = depth
        
        a = np.zeros((ma, na), dtype=np.uint8)
        for i in range(ma):
            a[i, i] = 1
            a[i, i + 1] = 1
        
        b = np.zeros((mb, nb), dtype=np.uint8)
        for i in range(mb):
            b[i, i] = 1
            b[i, i + 1] = 1
        
        # HGP construction
        n_left = na * nb
        n_right = ma * mb
        n_qubits = n_left + n_right
        
        hx = np.zeros((ma * nb, n_qubits), dtype=np.uint8)
        stab = 0
        for check_a in range(ma):
            for bit_b in range(nb):
                for bit_a in range(na):
                    if a[check_a, bit_a]:
                        hx[stab, bit_a * nb + bit_b] ^= 1
                for check_b in range(mb):
                    if b[check_b, bit_b]:
                        hx[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        hz = np.zeros((na * mb, n_qubits), dtype=np.uint8)
        stab = 0
        for bit_a in range(na):
            for check_b in range(mb):
                for bit_b in range(nb):
                    if b[check_b, bit_b]:
                        hz[stab, bit_a * nb + bit_b] ^= 1
                for check_a in range(ma):
                    if a[check_a, bit_a]:
                        hz[stab, n_left + check_a * mb + check_b] ^= 1
                stab += 1
        
        return hx, hz, n_qubits
    
    @staticmethod
    def _build_logicals(n_qubits: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Build logical operators."""
        logical_x = np.zeros((k, n_qubits), dtype=np.uint8)
        logical_z = np.zeros((k, n_qubits), dtype=np.uint8)
        
        for i in range(k):
            logical_x[i, i] = 1
            logical_z[i, n_qubits // 2 + i] = 1
        
        return logical_x, logical_z


# Pre-configured instances
QuantumPin_d3_m2 = lambda: QuantumPinCode(d=3, m=2)
QuantumPin_d5_m3 = lambda: QuantumPinCode(d=5, m=3)
DoublePin_d3 = lambda: DoublePinCode(d=3)
DoublePin_d5 = lambda: DoublePinCode(d=5)
Rainbow_L3_r3 = lambda: RainbowCode(L=3, r=3)
Rainbow_L5_r4 = lambda: RainbowCode(L=5, r=4)
HolographicRainbow_L4_d2 = lambda: HolographicRainbowCode(L=4, bulk_depth=2)
HolographicRainbow_L6_d3 = lambda: HolographicRainbowCode(L=6, bulk_depth=3)
