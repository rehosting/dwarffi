"""dwarffi package."""
try:
    from ._version import version as __version__
except Exception:
    __version__ = "0+unknown"

from .core import *  # noqa: F403
from .dffi import DFFI

__all__ = ["__version__", "DFFI"]