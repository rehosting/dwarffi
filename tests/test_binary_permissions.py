import os
import subprocess

import pytest

from dwarffi.utils import get_dwarf2json_path


def test_dwarf2json_is_executable():
    """
    Ensures that get_dwarf2json_path() successfully resolves a binary,
    and that the resolved file has the necessary executable permissions 
    to be run by the OS without throwing [Errno 13] Permission denied.
    """
    bin_path = get_dwarf2json_path()
    
    # 1. Ensure the function actually found something
    assert bin_path is not None, "dwarf2json binary was not found by get_dwarf2json_path()"
    
    # 2. Ensure the path exists on disk
    assert os.path.exists(bin_path), f"Binary path does not exist: {bin_path}"
    
    # 3. Ensure it is a file
    assert os.path.isfile(bin_path), f"Binary path is not a file: {bin_path}"
    
    # 4. Verify we have Execution permissions
    assert os.access(bin_path, os.X_OK), f"OS reports binary lacks executable permissions: {bin_path}"
    
    # 5. Sanity check: Actually try to invoke it to guarantee no silent OS-level blocks
    try:
        # Running the binary with no args or --help should exit cleanly
        result = subprocess.run([bin_path, "--help"], capture_output=True, text=True, check=False)
        assert result.returncode == 0, f"Binary failed to execute. Output: {result.stderr}"
    except PermissionError as e:
        pytest.fail(f"Subprocess threw PermissionError when trying to execute binary: {e}")
    except Exception as e:
        pytest.fail(f"Unexpected error running binary: {e}")