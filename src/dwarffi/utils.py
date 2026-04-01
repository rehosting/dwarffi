import importlib.resources
import os
from typing import Optional


def get_dwarf2json_path() -> Optional[str]:
    """Attempts to find the bundled dwarf2json binary."""
    try:
        # Assuming you put the binary in src/dwarffi/bin/dwarf2json
        with importlib.resources.path("dwarffi.bin", "dwarf2json") as p:
            if os.path.exists(p):
                return str(p)
    except (ImportError, FileNotFoundError):
        pass
    
    # Fallback to checking the system PATH just in case they installed it manually
    import shutil
    return shutil.which("dwarf2json")