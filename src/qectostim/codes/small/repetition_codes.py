"""
Repetition Code — [[N, 1, N]] One-Directional CSS Code

The repetition code is the simplest quantum error-correcting code and the
foundational building block for constructing topological codes via the
hypergraph product.

Overview
--------
The repetition code encodes **1 logical qubit** in **N physical qubits**
arranged in a linear chain.  It detects and corrects **bit-flip (X) errors**
using N−1 adjacent-pair Z-type parity checks, but offers **no protection
against phase-flip (Z) errors**.

This one-directional protection makes it a *classical* code embedded in
the CSS framework.  Despite this limitation, it is essential because:

1. It is the simplest code to understand syndrome extraction on.
2. It is the base case for Stim's ``gen_repetition_code`` circuits.
3. Its chain complex is the building block for hypergraph products:
   **Rep(L) ⊗ Rep(L) = Toric(L)**.

Code parameters
---------------
* **n** = N physical qubits
* **k** = 1 logical qubit
* **d** = N (distance for X errors; distance 1 for Z errors)
* **Rate** R = 1/N

Stabilisers
-----------
The code has N−1 weight-2 Z-type stabilisers, one per adjacent pair:

    Sᵢ = Zᵢ Zᵢ₊₁    for i = 0, 1, …, N−2

There are **no X-type stabilisers** — this is what makes the code
one-directional.  Equivalently, H_X is empty and H_Z is an (N−1)×N
bidiagonal matrix.

Logical operators
-----------------
* **X̄** = X⊗N (X on all N qubits) — the full-chain operator.
  Weight = N, giving distance N against Z-type detection.
* **Z̄** = Z₀ (Z on the first qubit only) — weight 1.
  This commutes with all Z-stabilisers (trivially, since Z·Z = I on
  shared qubits and Z·I = Z otherwise).

Qubit layout (example: N=5)
---------------------------
Data qubits on a 1D chain.  Z-stabilisers sit between adjacent data qubits.

::

    Data qubits:    0       1       2       3       4
                    ●───────●───────●───────●───────●
                        │       │       │       │
    Z-stabilisers:    [Z₀]    [Z₁]    [Z₂]    [Z₃]

    Data qubit coordinates:
      i: (i, 0)   for i = 0, 1, …, N−1

    Z-stabiliser coordinates (midpoint between adjacent data qubits):
      Zⱼ: (j + 0.5, 0)   for j = 0, 1, …, N−2

    No X-stabilisers (one-directional code).

Chain complex (2-term)
----------------------
    C₁ (qubits, N) —∂₁→ C₀ (checks, N−1)

where ∂₁ is the bidiagonal (N−1)×N matrix.  This is the simplest
non-trivial chain complex and the seed for all hypergraph products.

Code Parameters
~~~~~~~~~~~~~~~
:math:`[[n, k, d]] = [[N, 1, N]]` where:

- :math:`n = N` physical qubits (linear chain)
- :math:`k = 1` logical qubit
- :math:`d = N` for X errors; distance 1 for Z errors
- Rate :math:`k/n = 1/N`

Stabiliser Structure
~~~~~~~~~~~~~~~~~~~~
- **X-type stabilisers**: none (one-directional code; ``H_X`` is empty).
- **Z-type stabilisers**: :math:`N - 1` generators, each weight 2;
  ``ZᵢZᵢ₊₁`` for :math:`i = 0, 1, \ldots, N-2` (adjacent-pair parity).
- Measurement schedule: all :math:`N - 1` Z-stabilisers in a single
  parallel round; each is a depth-2 CNOT circuit.

Connections to other codes
--------------------------
* **Toric code**: Rep(L) ⊗ Rep(L) = ToricCode(L) via the hypergraph
  product (Tillich–Zémor construction).
* **4D toric code**: Toric ⊗ Toric = 4D tesseract code.
* **Surface code**: the repetition code is the 1D boundary of the
  2D surface code — surface code stabilisers project onto repetition
  code checks at the boundary.

References
----------
* Shor, "Scheme for reducing decoherence in quantum computer memory",
  Phys. Rev. A 52, R2493 (1995).
* Tillich & Zémor, "Quantum LDPC codes with positive rate and minimum
  distance proportional to n^{1/2}", ISIT 2009.  arXiv:0903.0566
* Error Correction Zoo: https://errorcorrectionzoo.org/c/quantum_repetition
* Wikipedia: https://en.wikipedia.org/wiki/Repetition_code
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from qectostim.codes.abstract_css import CSSCodeWithComplex, Coord2D
from qectostim.codes.abstract_code import PauliString
from qectostim.codes.complexes.css_complex import CSSChainComplex2
from qectostim.codes.utils import validate_css_code


class RepetitionCode(CSSCodeWithComplex):
    """[[N, 1, N]] Repetition code with 2-chain complex.

    Encodes 1 logical qubit in N physical qubits with distance N (for X errors).
    Uses a linear chain where stabilisers check adjacent-pair parity.

    The 2-chain complex C₁ —∂₁→ C₀ is the fundamental building block for
    hypergraph products:

    * Rep(L) ⊗ Rep(L) → ToricCode(L)  (3-chain)
    * ToricCode ⊗ ToricCode → 4D Tesseract code  (5-chain)

    Parameters
    ----------
    N : int
        Number of physical qubits (must be ≥ 3).
    metadata : dict, optional
        Extra metadata merged into the code's metadata dictionary.

    Attributes
    ----------
    n : int
        Number of physical qubits (= N).
    k : int
        Number of logical qubits (= 1).
    chain_complex : CSSChainComplex2
        The 2-chain complex defining the code structure.

    Examples
    --------
    >>> code = RepetitionCode(N=5)
    >>> code.n, code.k
    (5, 1)
    >>> code.chain_complex.boundary_maps  # (N-1) x N bidiagonal
    {1: array([[1, 1, 0, 0, 0],
               [0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0],
               [0, 0, 0, 1, 1]], dtype=uint8)}

    Notes
    -----
    The repetition code only corrects X errors.  Z errors are undetectable
    because there are no X-type stabilisers.  This makes it a *classical*
    code embedded in the quantum CSS framework.

    Despite this limitation, the repetition code is critical for:

    1. Teaching syndrome extraction and decoding.
    2. Benchmarking decoders on the simplest possible code.
    3. Building topological codes via the hypergraph product.

    See Also
    --------
    ToricCode33 : HGP of two repetition codes.
    SteaneCode713 : Smallest CSS code correcting all single-qubit errors.
    """
    
    def __init__(self, N: int = 3, metadata: Optional[Dict[str, Any]] = None):
        """Initialize [[N,1,N]] Repetition Code
        
        Parameters
        ----------
        N : int, default=3
            Code size. Must be >= 3.
        metadata : dict, optional
            Additional metadata to store about the code.

        Raises
        ------
        ValueError
            If ``N < 3``.
        """
        if N < 3:
            raise ValueError(f"N must be >= 3. Got N={N}")
        
        self._N = N
        
        # Build the 2-chain complex
        chain_complex = self._build_chain_complex(N)
        
        # Generate logical operators  
        logical_x, logical_z = self._generate_logical_operators(N)
        
        # Setup metadata
        # Generate linear chain coordinates
        data_coords = [(float(i), 0.0) for i in range(N)]
        # Z stabilizer coordinates (between adjacent qubits)
        z_stab_coords = [(float(i) + 0.5, 0.0) for i in range(N - 1)]
        
        meta: Dict[str, Any] = metadata or {}
        meta.update({
            # ── Code parameters ────────────────────────────────────
            "code_family": "repetition",
            "code_type": "repetition",
            "name": f"Repetition_{N}",
            "n": N,
            "k": 1,
            "distance": N,
            "x_distance": N,   # distance for X errors (full-chain)
            "z_distance": 1,   # distance for Z errors (undetectable)
            "rate": 1.0 / N,
            "code_size": N,
            "logical_qubits": 1,
            # ── Geometry ───────────────────────────────────────────
            "data_coords": data_coords,
            "x_stab_coords": [],  # No X stabilisers
            "z_stab_coords": z_stab_coords,
            "data_qubits": list(range(N)),
            "x_logical_coords": data_coords,        # full chain
            "z_logical_coords": [data_coords[0]],    # single qubit
            # ── Logical operator Pauli types ───────────────────────
            "lx_pauli_type": "X",
            "lx_support": list(range(N)),  # Full chain
            "lz_pauli_type": "Z",
            "lz_support": [0],  # Single qubit
            # ── Stabiliser scheduling ──────────────────────────────
            "stabiliser_schedule": {
                "x_rounds": {},  # No X stabilisers
                "z_rounds": {i: 0 for i in range(N - 1)},
                "n_rounds": 1,
                "description": (
                    "Fully parallel: all N−1 Z-stabilisers in round 0. "
                    "No X-stabilisers (one-directional code)."
                ),
            },
            "z_schedule": [(0.5, 0.0), (-0.5, 0.0)],  # Adjacent pair check
            "x_schedule": None,  # No X stabilisers (one-directional code)
            # ── Literature / provenance ────────────────────────────
            "error_correction_zoo_url": "https://errorcorrectionzoo.org/c/quantum_repetition",
            "wikipedia_url": "https://en.wikipedia.org/wiki/Repetition_code",
            "canonical_references": [
                "Shor, Phys. Rev. A 52, R2493 (1995)",
                "Tillich & Z\u00e9mor, ISIT 2009. arXiv:0903.0566",
            ],
            "connections": [
                "HGP building block: Rep(L) \u2297 Rep(L) = ToricCode(L)",
                "Simplest Stim circuit: gen_repetition_code",
                "1D boundary of the 2D surface code",
            ],
        })
        
        # ── Validate CSS structure ─────────────────────────────────
        # Repetition code has empty Hx (no X-stabilisers), so the CSS
        # commutativity check is trivially satisfied.
        hz_mat = chain_complex.boundary_maps[1]
        hx_mat = np.zeros((0, N), dtype=np.uint8)
        validate_css_code(hx_mat, hz_mat, f"Repetition_{N}", raise_on_error=True)

        # Call parent CSSCodeWithComplex constructor
        super().__init__(
            chain_complex=chain_complex,
            logical_x=logical_x,
            logical_z=logical_z,
            metadata=meta,
        )
    
    @property
    def N(self) -> int:
        """Code size (number of physical qubits)."""
        return self._N

    @property
    def distance(self) -> int:
        """Code distance (= N for X errors)."""
        return self._N

    @property
    def name(self) -> str:
        """Human-readable name, e.g. ``'RepetitionCode(5)'``."""
        return f"RepetitionCode({self._N})"
    
    @staticmethod
    def _build_chain_complex(N: int) -> CSSChainComplex2:
        """Build the 2-chain complex for the repetition code.
        
        The chain complex is:
            C1 (N qubits) --∂1--> C0 (N-1 checks)
        
        where ∂1[i, j] = 1 if qubit j is in check i.
        For adjacent-pair checks: ∂1[i, i] = ∂1[i, i+1] = 1.
        
        Parameters
        ----------
        N : int
            Number of qubits
            
        Returns
        -------
        CSSChainComplex2
            The 2-chain complex with boundary_1 = Hz
        """
        # boundary_1: (N-1) × N matrix
        # Each row i has 1s at positions i and i+1
        boundary_1 = np.zeros((N - 1, N), dtype=np.uint8)
        for i in range(N - 1):
            boundary_1[i, i] = 1
            boundary_1[i, i + 1] = 1
        
        return CSSChainComplex2(boundary_1=boundary_1)
    
    @staticmethod
    def _generate_logical_operators(N: int) -> Tuple[List[PauliString], List[PauliString]]:
        """Generate logical X and Z operators
        
        For the [[N,1,N]] repetition code:
        
        Lx = [X,X,...,X] (full-chain X):
        - Full-chain X anticommutes with most stabilizer measurements at boundaries
        - Encodes logical information: parity of entire chain
        - Distance N for detecting/correcting X errors
        
        Lz = [Z,I,...,I] (single-qubit Z):
        - Single qubit Z commutes with all Hz checks (even overlap with each pair)
        - Represents storable information in one qubit's Z-basis state
        
        Returns
        -------
        Lx : list[PauliString]
            Logical X operators (1 for [[N,1,N]])
        Lz : list[PauliString]
            Logical Z operators (1 for [[N,1,N]])
        """
        # Full-chain X operator
        lx_str = "X" * N
        
        # Single-qubit Z operator (first qubit)
        lz_str = "Z" + "I" * (N - 1)
        
        return [lx_str], [lz_str]
    
    @property
    def chain_complex(self) -> CSSChainComplex2:
        """The underlying 2-chain complex."""
        return self._chain_complex
    
    def qubit_coords(self) -> List[Coord2D]:
        """Return qubit coordinates for visualization (linear chain)."""
        return list(self._metadata.get("data_coords", []))


# Convenience factory functions for common code sizes
def create_repetition_code_3() -> RepetitionCode:
    """Create a [[3, 1, 3]] repetition code.

    The smallest non-trivial repetition code.  Its 2-chain complex
    is the seed for the hypergraph product: ``Rep(3) ⊗ Rep(3) → Toric(3)``.

    Returns
    -------
    RepetitionCode
        A [[3, 1, 3]] code instance.
    """
    return RepetitionCode(N=3)


def create_repetition_code_5() -> RepetitionCode:
    """Create a [[5, 1, 5]] repetition code.

    Corrects up to 2 bit-flip errors.  Common benchmark size for
    Stim's ``gen_repetition_code`` circuits.

    Returns
    -------
    RepetitionCode
        A [[5, 1, 5]] code instance.
    """
    return RepetitionCode(N=5)


def create_repetition_code_7() -> RepetitionCode:
    """Create a [[7, 1, 7]] repetition code.

    Corrects up to 3 bit-flip errors.  Same qubit count as the
    Steane code but with one-directional protection only.

    Returns
    -------
    RepetitionCode
        A [[7, 1, 7]] code instance.
    """
    return RepetitionCode(N=7)


def create_repetition_code_9() -> RepetitionCode:
    """Create a [[9, 1, 9]] repetition code.

    Corrects up to 4 bit-flip errors.  Same qubit count as the
    Shor code but with one-directional protection only.

    Returns
    -------
    RepetitionCode
        A [[9, 1, 9]] code instance.
    """
    return RepetitionCode(N=9)

