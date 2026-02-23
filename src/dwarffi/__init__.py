"""dwarffi package."""
try:
    from ._version import version as __version__
except Exception:
    __version__ = "0+unknown"

__all__ = ["__version__"]
from .dffi import DFFI
from .core import *
