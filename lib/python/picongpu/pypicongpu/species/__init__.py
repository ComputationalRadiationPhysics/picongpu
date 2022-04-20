"""
Data structures to specify and initialize particle species.

Note that the data structure (the classes) here use a different architecure
than both (!) PIConGPU and PICMI.

Please refer to the documentation for a deeper discussion.
"""

from . import operation
from . import attribute
from . import constant

from .species import Species
from .initmanager import InitManager

__all__ = [
    "Species",
    "InitManager",
    "attribute",
    "constant",
    "operation",
]
