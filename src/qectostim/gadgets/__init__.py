# src/qectostim/gadgets/__init__.py
"""
Fault-tolerant logical gate gadgets for quantum error correction.

This package provides implementations of fault-tolerant logical gates
for QEC codes, including:

1. **Transversal Gates** (`transversal.py`):
   - Single-qubit: H, S, T, X, Y, Z
   - Two-qubit: CNOT, CZ, SWAP between code blocks
   - Maximum parallelism, simplest fault-tolerance

2. **Teleportation-Based Gates** (`teleportation.py`):
   - Clifford gates via teleportation protocol
   - Magic state injection for T gates
   - Works on any CSS code

3. **CSS Code Surgery** (`css_surgery.py`):
   - Lattice surgery merge/split operations
   - Universal CNOT between CSS code patches
   - ZZ and XX merge operations

4. **Infrastructure Modules**:
   - `coordinates.py`: N-dimensional coordinate utilities
   - `layout.py`: Multi-code spatial layout management
   - `scheduling.py`: Parallel gate scheduling
   - `pauli_frame.py`: Pauli frame tracking
   - `base.py`: Abstract base classes and mixins

Example usage:
    >>> from qectostim.gadgets import TransversalHadamard, SurgeryCNOT
    >>> from qectostim.codes.surface import SurfaceCode
    >>> from qectostim.noise.models import UniformDepolarizing
    >>>
    >>> code = SurfaceCode(distance=3)
    >>> noise = UniformDepolarizing(p=0.001)
    >>> 
    >>> # Single-qubit transversal H
    >>> h_gadget = TransversalHadamard()
    >>> h_circuit = h_gadget.to_stim([code], noise)
    >>>
    >>> # Two-code surgery CNOT
    >>> code1, code2 = SurfaceCode(distance=3), SurfaceCode(distance=3)
    >>> cnot_gadget = SurgeryCNOT()
    >>> cnot_circuit = cnot_gadget.to_stim([code1, code2], noise)
"""

# Base classes and metadata
from qectostim.gadgets.base import (
    Gadget,
    GadgetMetadata,
    StabilizerTransform,
    ObservableTransform,
    TwoQubitObservableTransform,
    TransversalGadgetMixin,
    SurgeryGadgetMixin,
    TeleportationGadgetMixin,
    PhaseResult,
    PhaseType,
)

# Coordinate utilities
from qectostim.gadgets.coordinates import (
    CoordND,
    get_code_dimension,
    get_bounding_box,
    translate_coords,
    pad_coord_to_dim,
    emit_qubit_coords_nd,
    emit_detector_nd,
)

# Layout management
from qectostim.gadgets.layout import (
    GadgetLayout,
    BlockInfo,
    BridgeAncilla,
    QubitIndexMap,
    QubitAllocation,
    BlockAllocation,
)

# Scheduling
from qectostim.gadgets.scheduling import (
    GadgetScheduler,
    CircuitLayer,
)

# Transversal gates
from qectostim.gadgets.transversal import (
    TransversalGate,
    TransversalHadamard,
    TransversalS,
    TransversalSDag,
    TransversalT,
    TransversalTDag,
    TransversalX,
    TransversalY,
    TransversalZ,
    TransversalCNOT,
    TransversalCZ,
    TransversalSWAP,
    get_transversal_gadget,
)

# Teleportation-based gates
from qectostim.gadgets.teleportation import (
    TeleportedGate,
    BellStateTeleportedGate,
    TeleportedHadamard,
    TeleportedS,
    TeleportedSDag,
    TeleportedT,
    TeleportedIdentity,
    BellTeleportedS,
    BellTeleportedSDag,
    TeleportationProtocol,
    AncillaState,
    get_teleported_gadget,
)

# CSS surgery
from qectostim.gadgets.css_surgery import (
    SurgeryType,
    SurgeryBoundary,
    LatticeZZMerge,
    LatticeXXMerge,
    SurgeryCNOT,
    get_surgery_gadget,
)

# Pauli frame tracking
from qectostim.gadgets.pauli_frame import (
    PauliFrame,
    PauliTracker,
    PauliType,
    pauli_product,
)


__all__ = [
    # Base
    "Gadget",
    "GadgetMetadata",
    "StabilizerTransform",
    "ObservableTransform",
    "TwoQubitObservableTransform",
    "TransversalGadgetMixin",
    "SurgeryGadgetMixin",
    "TeleportationGadgetMixin",
    "PhaseResult",
    "PhaseType",
    # Coordinates
    "CoordND",
    "get_code_dimension",
    "get_bounding_box",
    "translate_coords",
    "pad_coord_to_dim",
    "emit_qubit_coords_nd",
    "emit_detector_nd",
    # Layout
    "GadgetLayout",
    "BlockInfo",
    "BridgeAncilla",
    "QubitIndexMap",
    "QubitAllocation",
    "BlockAllocation",
    # Scheduling
    "GadgetScheduler",
    "CircuitLayer",
    # Transversal
    "TransversalGate",
    "TransversalHadamard",
    "TransversalS",
    "TransversalSDag",
    "TransversalT",
    "TransversalTDag",
    "TransversalX",
    "TransversalY",
    "TransversalZ",
    "TransversalCNOT",
    "TransversalCZ",
    "TransversalSWAP",
    "get_transversal_gadget",
    # Teleportation
    "TeleportedGate",
    "BellStateTeleportedGate",
    "TeleportedHadamard",
    "TeleportedS",
    "TeleportedSDag",
    "TeleportedT",
    "TeleportedIdentity",
    "BellTeleportedS",
    "BellTeleportedSDag",
    "TeleportationProtocol",
    "AncillaState",
    "get_teleported_gadget",
    # Surgery
    "SurgeryType",
    "SurgeryBoundary",
    "LatticeZZMerge",
    "LatticeXXMerge",
    "SurgeryCNOT",
    "get_surgery_gadget",
    # Pauli frame
    "PauliFrame",
    "PauliTracker",
    "PauliType",
    "pauli_product",
]
