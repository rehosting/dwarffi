# src/dwarffi/cli.py
import sys
import subprocess
from pathlib import Path

def main() -> None:
    """Entry point for the dwarf2json command line wrapper."""
    # Resolve the path to the bundled binary relative to this Python file
    package_dir = Path(__file__).parent
    
    # Handle the `.exe` extension for Windows environments
    exe_name = "dwarf2json.exe" if sys.platform == "win32" else "dwarf2json"
    binary_path = package_dir / "bin" / exe_name
    
    if not binary_path.exists():
        print(f"Error: Bundled dwarf2json binary not found at {binary_path}", file=sys.stderr)
        sys.exit(1)
        
    # Execute the binary, passing all arguments provided by the user
    try:
        result = subprocess.run([str(binary_path)] + sys.argv[1:])
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(130)