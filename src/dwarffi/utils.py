import importlib.resources
import os
import stat
from typing import Optional


def get_dwarf2json_path() -> Optional[str]:
    """Attempts to find the bundled dwarf2json binary."""
    try:
        # Resolves the absolute path inside the user's site-packages
        with importlib.resources.path("dwarffi.bin", "dwarf2json") as p:
            if os.path.exists(p):
                # Ensure the executable bit survived the PyPI unpacking
                try:
                    st = os.stat(p)
                    os.chmod(p, st.st_mode | stat.S_IEXEC)
                except (PermissionError, OSError):
                    # Ignore if we don't have permission to chmod (e.g., system-wide install)
                    pass
                return str(p)
                
        # Also check for Windows .exe extension
        with importlib.resources.path("dwarffi.bin", "dwarf2json.exe") as p:
            if os.path.exists(p):
                return str(p)
    except (ImportError, FileNotFoundError):
        pass
    
    # Fallback to checking the system PATH just in case they installed it manually
    import shutil
    return shutil.which("dwarf2json")