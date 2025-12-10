# src/qectostim/codes/base/__init__.py
"""
Backward compatibility re-exports for code classes.

This module re-exports all concrete code implementations from their
category-specific subfolders (small/, surface/, color/, etc.).
"""

# Re-export from category folders for backward compatibility
try:
    from ..small import *
except ImportError:
    pass

try:
    from ..surface import *
except ImportError:
    pass

try:
    from ..color import *
except ImportError:
    pass

try:
    from ..qldpc import *
except ImportError:
    pass

try:
    from ..subsystem import *
except ImportError:
    pass

try:
    from ..floquet import *
except ImportError:
    pass

try:
    from ..topological import *
except ImportError:
    pass

try:
    from ..generic import *
except ImportError:
    pass

try:
    from ..qudit import *
except ImportError:
    pass

try:
    from ..bosonic import *
except ImportError:
    pass
