import json
import pytest
from unittest import mock
from dwarffi.dffi import DFFI

def mock_subprocess_run(cmd, **kwargs):
    """Mocks the subprocess runner to simulate GCC and dwarf2json."""
    class MockCompletedProcess:
        def __init__(self, stdout):
            self.stdout = stdout
            
    if cmd[0] == "dwarf2json":
        # Simulate dwarf2json returning a valid ISF definition
        fake_isf = {
            "metadata": {},
            "base_types": {
                "custom_int": {"kind": "int", "size": 4, "signed": True, "endian": "little"}
            },
            "user_types": {}, "enums": {}, "symbols": {}, "typedefs": {}
        }
        return MockCompletedProcess(json.dumps(fake_isf))
        
    if cmd[0].endswith("gcc"):
        # Simulate successful compilation
        return MockCompletedProcess("")

    raise ValueError(f"Unexpected command: {cmd}")


def test_cdef_success(tmp_path):
    ffi = DFFI()
    
    # Mock shutil.which to pretend our tools exist in PATH, 
    # and subprocess to mock compilation output.
    with mock.patch("shutil.which", return_value="/usr/bin/mocked_tool"):
        with mock.patch("subprocess.run", side_effect=mock_subprocess_run):
            
            # Test compiling some arbitrary C code and saving it to an XZ file
            out_file = tmp_path / "types.json.xz"
            ffi.cdef(
                "typedef int custom_int;", 
                compiler="arm-none-eabi-gcc", 
                compiler_flags=["-g", "-c", "-mthumb"],
                save_isf_to=str(out_file)
            )
            
    # The ISF dictionary returned by our mock should be successfully loaded
    assert ffi.sizeof("custom_int") == 4
    
    # The resulting file should have been written and compressed
    assert out_file.exists()
    
    # Let's verify we can load the newly generated compressed file!
    ffi_new = DFFI(str(out_file))
    assert ffi_new.sizeof("custom_int") == 4


def test_cdef_missing_dwarf2json():
    ffi = DFFI()
    
    # Simulate dwarf2json missing from the system
    def mock_which(cmd):
        if "dwarf2json" in cmd:
            return None
        return "/usr/bin/gcc"
        
    with mock.patch("shutil.which", side_effect=mock_which):
        with pytest.raises(RuntimeError, match="dwarf2json.*not found in PATH"):
            ffi.cdef("int a;")


def test_cdef_missing_compiler():
    ffi = DFFI()
    
    def mock_which(cmd):
        if "gcc" in cmd:
            return None
        return "/usr/bin/dwarf2json"
        
    with mock.patch("shutil.which", side_effect=mock_which):
        with pytest.raises(RuntimeError, match="Compiler 'gcc' not found in PATH"):
            ffi.cdef("int a;")