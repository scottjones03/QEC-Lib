from typing import Dict, Tuple
import numpy as np


def kron_gf2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Kronecker product over GF(2)."""
    return (np.kron(a, b).astype(np.uint8) % 2)


class ChainComplex:
    """
    Generic chain complex

        ...  C_k --sigma_k--> C_{k-1} -- ... -- C_0

    Represented by:
        - boundary_maps[k] = sigma_k as a numpy array.
          sigma_k has shape (dim C_{k-1}, dim C_k),
          i.e. columns index basis vectors of C_k.

    Attributes:
        boundary_maps: dict mapping grade k to the boundary map sigma_k.
        max_grade: highest k present (e.g. 2 for a 3-term complex C2->C1->C0,
                   4 for a 5-term complex C4->C3->C2->C1->C0).
        qubit_grade: the chain group C_{qubit_grade} that hosts the *data qubits*.
                     (For 2D surface code, qubits on edges → grade 1.
                      For 4D surface code, qubits on 2-cells → grade 2.)
    """

    def __init__(self, boundary_maps: Dict[int, np.ndarray], qubit_grade: int):
        if not boundary_maps:
            raise ValueError("boundary_maps must be a non-empty dict.")

        self.boundary_maps: Dict[int, np.ndarray] = boundary_maps
        self.max_grade: int = max(boundary_maps.keys())
        self.qubit_grade: int = qubit_grade

        # Optional sanity check: sigma_{k-1} @ sigma_k = 0 over Z2
        # (you can comment this out if performance becomes an issue).
        for k, sigma_k in boundary_maps.items():
            prev_k = k - 1
            if prev_k in boundary_maps:
                sigma_prev = boundary_maps[prev_k]
                comp = (sigma_prev @ sigma_k) % 2
                if np.any(comp):
                    raise ValueError(
                        f"Chain condition violated: sigma_{prev_k} * sigma_{k} != 0 over Z2."
                    )

    def boundary(self, k: int) -> np.ndarray:
        """Return sigma_k: C_k -> C_{k-1}."""
        return self.boundary_maps[k]
    
    def dim(self, k: int) -> int:
        """Return the dimension of C_k (number of k-cells).
        
        For grade k, the dimension is:
        - If k in boundary_maps: number of columns of boundary_maps[k]
        - If k+1 in boundary_maps: number of rows of boundary_maps[k+1]
        - Otherwise: 0
        """
        if k in self.boundary_maps:
            return self.boundary_maps[k].shape[1]
        elif k + 1 in self.boundary_maps:
            return self.boundary_maps[k + 1].shape[0]
        return 0
    
    @property
    def min_grade(self) -> int:
        """Minimum grade present (typically 0 or 1)."""
        # Find the lowest grade that has cells
        grades = set(self.boundary_maps.keys())
        # Also include grade 0 if grade 1 is present
        min_k = min(grades)
        return min_k - 1 if min_k > 0 else 0


def tensor_product_chain_complex(
    A: ChainComplex,
    B: ChainComplex,
) -> ChainComplex:
    """
    Compute the tensor product of two chain complexes over GF(2).
    
    Given chain complexes A and B:
        A: ... → A_i → A_{i-1} → ...
        B: ... → B_j → B_{j-1} → ...
    
    The tensor product (A ⊗ B) has:
        (A ⊗ B)_k = ⊕_{i+j=k} A_i ⊗ B_j
    
    The boundary map on A_i ⊗ B_j is:
        ∂^{A⊗B}(a ⊗ b) = (∂^A a) ⊗ b + (-1)^i a ⊗ (∂^B b)
    
    Over GF(2), the sign (-1)^i = 1, so:
        ∂^{A⊗B} = ∂^A ⊗ I_B + I_A ⊗ ∂^B
    
    This is the hypergraph product construction at the chain complex level.
    
    Chain length of product:
        max_grade(A ⊗ B) = max_grade(A) + max_grade(B)
    
    Examples:
        - 2-chain ⊗ 2-chain → 3-chain (RepetitionCode ⊗ RepetitionCode → ToricCode)
        - 3-chain ⊗ 3-chain → 5-chain (ToricCode ⊗ ToricCode → 4D Tesseract)
    
    Parameters
    ----------
    A : ChainComplex
        First chain complex
    B : ChainComplex
        Second chain complex
        
    Returns
    -------
    ChainComplex
        The tensor product chain complex with:
        - max_grade = A.max_grade + B.max_grade
        - qubit_grade = A.qubit_grade + B.qubit_grade
    """
    # Compute dimensions of each grade in A and B
    def get_dim(cc: ChainComplex, k: int) -> int:
        """Get dimension of C_k."""
        if k in cc.boundary_maps:
            return cc.boundary_maps[k].shape[1]
        elif k + 1 in cc.boundary_maps:
            return cc.boundary_maps[k + 1].shape[0]
        elif k - 1 in cc.boundary_maps:
            # Grade 0 case: rows of boundary_1
            return cc.boundary_maps[k].shape[0] if k in cc.boundary_maps else 0
        return 0
    
    # Get all grades present in each complex
    a_grades = set()
    b_grades = set()
    
    for k in A.boundary_maps:
        a_grades.add(k)
        a_grades.add(k - 1)  # Target of boundary map
    
    for k in B.boundary_maps:
        b_grades.add(k)
        b_grades.add(k - 1)
    
    # Remove negative grades
    a_grades = {k for k in a_grades if k >= 0}
    b_grades = {k for k in b_grades if k >= 0}
    
    # Compute dimensions
    a_dims = {}
    b_dims = {}
    
    for k in a_grades:
        if k in A.boundary_maps:
            a_dims[k] = A.boundary_maps[k].shape[1]
        elif k + 1 in A.boundary_maps:
            a_dims[k] = A.boundary_maps[k + 1].shape[0]
    
    for k in b_grades:
        if k in B.boundary_maps:
            b_dims[k] = B.boundary_maps[k].shape[1]
        elif k + 1 in B.boundary_maps:
            b_dims[k] = B.boundary_maps[k + 1].shape[0]
    
    # Product chain: (A ⊗ B)_k = ⊕_{i+j=k} A_i ⊗ B_j
    # We need to build boundary maps for the product
    
    max_grade_product = A.max_grade + B.max_grade
    
    # Compute dimensions of each grade in the product
    product_dims = {}
    for k in range(max_grade_product + 1):
        dim_k = 0
        for i in range(k + 1):
            j = k - i
            if i in a_dims and j in b_dims:
                dim_k += a_dims[i] * b_dims[j]
        product_dims[k] = dim_k
    
    # Build boundary maps for the product
    # ∂_k^{A⊗B}: (A⊗B)_k → (A⊗B)_{k-1}
    product_boundary_maps = {}
    
    for k in range(1, max_grade_product + 1):
        # Source: (A⊗B)_k = ⊕_{i+j=k} A_i ⊗ B_j
        # Target: (A⊗B)_{k-1} = ⊕_{p+q=k-1} A_p ⊗ B_q
        
        source_dim = product_dims.get(k, 0)
        target_dim = product_dims.get(k - 1, 0)
        
        if source_dim == 0 or target_dim == 0:
            continue
        
        # Build the boundary map as a block matrix
        boundary_k = np.zeros((target_dim, source_dim), dtype=np.uint8)
        
        # Track column and row offsets for each (i,j) and (p,q) block
        source_blocks = []  # List of (i, j, col_start, col_end)
        for i in range(k + 1):
            j = k - i
            if i in a_dims and j in b_dims:
                source_blocks.append((i, j, a_dims[i], b_dims[j]))
        
        target_blocks = []  # List of (p, q, row_start, row_end)
        for p in range(k):
            q = (k - 1) - p
            if p in a_dims and q in b_dims:
                target_blocks.append((p, q, a_dims[p], b_dims[q]))
        
        # Compute column offsets for source blocks
        source_offsets = {}
        col_offset = 0
        for i, j, dim_ai, dim_bj in source_blocks:
            source_offsets[(i, j)] = col_offset
            col_offset += dim_ai * dim_bj
        
        # Compute row offsets for target blocks
        target_offsets = {}
        row_offset = 0
        for p, q, dim_ap, dim_bq in target_blocks:
            target_offsets[(p, q)] = row_offset
            row_offset += dim_ap * dim_bq
        
        # Fill in the boundary map
        # For each source block (i, j), contribute to target blocks:
        # - (i-1, j) via ∂^A ⊗ I_B (if i > 0)
        # - (i, j-1) via I_A ⊗ ∂^B (if j > 0)
        
        for i, j, dim_ai, dim_bj in source_blocks:
            col_start = source_offsets[(i, j)]
            block_cols = dim_ai * dim_bj
            
            # Contribution from ∂^A ⊗ I_B to block (i-1, j)
            if i > 0 and (i - 1, j) in target_offsets:
                if i in A.boundary_maps:
                    dA = A.boundary_maps[i]  # shape (dim A_{i-1}, dim A_i)
                    I_Bj = np.eye(dim_bj, dtype=np.uint8)
                    # ∂^A ⊗ I_B has shape (dim A_{i-1} * dim B_j, dim A_i * dim B_j)
                    block = kron_gf2(dA, I_Bj)
                    
                    row_start = target_offsets[(i - 1, j)]
                    target_rows = a_dims[i - 1] * dim_bj
                    
                    boundary_k[row_start:row_start + target_rows, col_start:col_start + block_cols] ^= block
            
            # Contribution from I_A ⊗ ∂^B to block (i, j-1)
            if j > 0 and (i, j - 1) in target_offsets:
                if j in B.boundary_maps:
                    I_Ai = np.eye(dim_ai, dtype=np.uint8)
                    dB = B.boundary_maps[j]  # shape (dim B_{j-1}, dim B_j)
                    # I_A ⊗ ∂^B has shape (dim A_i * dim B_{j-1}, dim A_i * dim B_j)
                    block = kron_gf2(I_Ai, dB)
                    
                    row_start = target_offsets[(i, j - 1)]
                    target_rows = dim_ai * b_dims[j - 1]
                    
                    boundary_k[row_start:row_start + target_rows, col_start:col_start + block_cols] ^= block
        
        if boundary_k.size > 0:
            product_boundary_maps[k] = boundary_k
    
    # Qubit grade for tensor product of chain complexes.
    # CSS codes place qubits on the "middle" chain group:
    #   - 3-chain (C2→C1→C0): qubits on C1, max_grade=2, qubit_grade=1
    #   - 5-chain (C4→C3→C2→C1→C0): qubits on C2, max_grade=4, qubit_grade=2
    #
    # Formula: qubit_grade = max_grade_product // 2
    # This ensures Hx from boundary above and Hz from boundary at qubit grade.
    product_qubit_grade = max_grade_product // 2
    
    return ChainComplex(
        boundary_maps=product_boundary_maps,
        qubit_grade=product_qubit_grade,
    )