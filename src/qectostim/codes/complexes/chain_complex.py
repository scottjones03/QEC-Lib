from typing import Dict
import numpy as np


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