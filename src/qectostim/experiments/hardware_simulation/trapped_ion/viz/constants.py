"""
Styling constants for QCCD visualization.

Tuned for publication-quality matplotlib output.  Imported by all
other modules in the ``viz`` subpackage.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

try:
    import matplotlib.patheffects as path_effects
    _HAS_PE = True
except ImportError:
    _HAS_PE = False

# =============================================================================
# General
# =============================================================================

DPI = 120
FONT_SIZE = 18
ION_FONT = 17           # role letter *inside* ion circle
ION_BELOW_FONT = 13     # index label *below* ion circle
TRAP_LABEL_FONT = 15
JUNCTION_FONT = 14
LEGEND_FONT = 14
GATE_FONT = 17
INFO_FONT = 13

# =============================================================================
# Geometry
# =============================================================================

ION_RADIUS = 0.42
JUNCTION_SIDE = 0.62
TRAP_PAD_X = 0.75
TRAP_PAD_Y = 0.60
ION_MARKER_SIZE = 800
SPACING = 3.8
ION_SPACING_RATIO = 0.92

# =============================================================================
# Line widths
# =============================================================================

TRAP_LINEWIDTH = 2.4
EDGE_LINEWIDTH = 2.4
ION_EDGE_WIDTH = 2.0
CROSSING_LW = 2.0
HIGHLIGHT_LW = 2.8

# =============================================================================
# Colour palette  (colour-blind friendly)
# =============================================================================

DATA_COLOR = "#2979FF"       # vivid blue
ANCILLA_COLOR = "#E53935"    # vivid red
SPECTATOR_COLOR = "#9E9E9E"  # grey
COOLING_COLOR = "#43A047"    # green
HIGHLIGHT_ION = "#FFD600"
HIGHLIGHT_BG = "#FFFDE7"

QUBIT_DEFAULT = DATA_COLOR
JUNCTION_FILL = "#FFA726"
JUNCTION_EDGE = "#E65100"
TRAP_FILL = "#E8EAF6"
TRAP_EDGE = "#5C6BC0"
TRAP_GATE_FILL = "#FFF9C4"
TRAP_GATE_EDGE = "#F9A825"

# --- Consistent rendering geometry -----------------------------------------

TRAP_PAD_ROUND = 0.18    # FancyBboxPatch boxstyle round pad
TRAP_LW = 2.2            # Trap border linewidth
JUNCTION_PAD_ROUND = 0.06
JUNCTION_LW = 1.2
CROSSING_COLOR = "#78909C"
CROSSING_VERT_COLOR = "#B0BEC5"
ION_OUTLINE = "#212121"

# --- role → colour mapping ------------------------------------------------

_ROLE_COLORS: Dict[str, str] = {
    "D": DATA_COLOR, "data": DATA_COLOR,
    "M": ANCILLA_COLOR, "ancilla": ANCILLA_COLOR, "A": ANCILLA_COLOR,
    "P": SPECTATOR_COLOR, "spectator": SPECTATOR_COLOR, "S": SPECTATOR_COLOR,
    "C": COOLING_COLOR, "cooling": COOLING_COLOR,
}

_ROLE_LEGEND: List[Tuple[str, str, str]] = [
    ("D", "Data qubit", DATA_COLOR),
    ("M", "Ancilla / Meas", ANCILLA_COLOR),
    ("P", "Placeholder (unused)", SPECTATOR_COLOR),
    ("C", "Cooling ion", COOLING_COLOR),
]

# --- Laser beam colours ---------------------------------------------------

LASER_MS = "#FFD600"       # yellow for MS / two-qubit
LASER_ROTATION = "#AB47BC" # purple for rotations
LASER_MEASURE = "#66BB6A"  # green for measurement
LASER_RESET = "#00BCD4"    # cyan for reset / initialisation

# --- Background -----------------------------------------------------------

BG_COLOR = "#FAFBFE"
SIDEBAR_BG = "#F8F8FC"

# --- Progress bar ----------------------------------------------------------

PROGRESS_BAR_COLOR = "#4CAF50"
PROGRESS_BAR_BG = "#E0E0E0"
PROGRESS_BAR_EDGE = "#BDBDBD"

# --- Physical timing defaults (microseconds) ------------------------------

H_PASS_TIME_US = 212.0
V_PASS_TIME_US = 510.0
GATE_TIME_US: Dict[str, float] = {
    "ms": 40.0,
    "rotation": 5.0,
    "measure": 100.0,
    "reset": 5.0,
}

# --- Gate-kind colour map (for sidebar dots) --------------------------------

GATE_KIND_COLORS: Dict[str, str] = {
    "ms": LASER_MS,
    "rotation": LASER_ROTATION,
    "measure": LASER_MEASURE,
    "reset": LASER_RESET,
}

# --- Background tints during gate steps -----------------------------------

GATE_BG_TINTS: Dict[str, Tuple[str, float]] = {
    "ms": (LASER_MS, 0.18),
    "rotation": (LASER_ROTATION, 0.15),
    "measure": (LASER_MEASURE, 0.15),
    "reset": (LASER_RESET, 0.10),
}

# =============================================================================
# Path effects (white strokes for text readability)
# =============================================================================

if _HAS_PE:
    STROKE = [path_effects.withStroke(linewidth=3.5, foreground="white")]
    STROKE_THIN = [path_effects.withStroke(linewidth=2.5, foreground="white")]
else:
    STROKE: list = []       # type: ignore[no-redef]
    STROKE_THIN: list = []  # type: ignore[no-redef]


# =============================================================================
# Operation class sets for gate-kind / transport classification
# =============================================================================

_OLD_TRANSPORT_CLASSES = {
    "Move", "JunctionCrossing", "Split", "Merge",
    "ReconfigurationStep", "GlobalReconfigurations",
    "ReconfigurationPlanner", "GlobalReconfiguration",
    "CrystalRotation", "SympatheticCooling", "CoolingOperation",
    "_EdgeOp", "TransportOperation",
}
_OLD_MS_CLASSES = {"TwoQubitMSGate", "MSGate", "TwoQubitGate", "GateSwap"}
_OLD_1Q_CLASSES = {"OneQubitGate", "XRotation", "YRotation", "SingleQubitGate"}
_OLD_MEAS_CLASSES = {"Measurement", "MeasurementOperation"}
_OLD_RESET_CLASSES = {"QubitReset", "ResetOperation"}
