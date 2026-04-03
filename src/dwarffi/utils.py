import importlib.resources
import os
import shutil
import stat
import tempfile
from typing import Optional


def get_dwarf2json_path() -> Optional[str]:
    """Attempts to find the bundled dwarf2json binary."""
    try:
        # Resolves the absolute path inside the user's site-packages
        with importlib.resources.path("dwarffi.bin", "dwarf2json") as p:
            if os.path.exists(p):
                # Check if it's already executable
                if os.access(p, os.X_OK):
                    return str(p)
                
                # Try to make it executable in-place
                try:
                    st = os.stat(p)
                    os.chmod(p, st.st_mode | stat.S_IEXEC)
                    return str(p)
                except (PermissionError, OSError):
                    # We lack permissions to chmod the system file. 
                    # Copy to a temporary directory where we have write access.
                    temp_bin = os.path.join(tempfile.gettempdir(), "dwarf2json_executable")
                    
                    # Only copy if we haven't already copied it in a previous run
                    if not os.path.exists(temp_bin) or not os.access(temp_bin, os.X_OK):
                        shutil.copy2(p, temp_bin)
                        st = os.stat(temp_bin)
                        os.chmod(temp_bin, st.st_mode | stat.S_IEXEC)
                        
                    return temp_bin
                
        # Also check for Windows .exe extension
        with importlib.resources.path("dwarffi.bin", "dwarf2json.exe") as p:
            if os.path.exists(p):
                return str(p)
    except (ImportError, FileNotFoundError):
        pass
    
    # Fallback to checking the system PATH just in case they installed it manually
    return shutil.which("dwarf2json")