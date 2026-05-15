import json
import os
import subprocess
from unittest import mock

import pytest

from dwarffi import DFFI


def test_load_elf_success(tmp_path):
    ffi = DFFI()

    def mock_subprocess_run(cmd, **kwargs):
        class MockCompletedProcess:
            def __init__(self, stdout):
                self.stdout = stdout

        if "dwarf2json" in cmd[0]:
            fake_isf = {
                "metadata": {},
                "base_types": {
                    "custom_int": {"kind": "int", "size": 4, "signed": True, "endian": "little"}
                },
                "user_types": {},
                "enums": {},
                "symbols": {},
                "typedefs": {},
            }
            return MockCompletedProcess(json.dumps(fake_isf))

        raise ValueError(f"Unexpected command: {cmd}")

    # Create a dummy ELF file on disk to pass the os.path.exists check
    fake_elf = tmp_path / "test_binary.elf"
    fake_elf.write_bytes(b"\x7fELF...")

    out_file = tmp_path / "types.json.xz"

    with mock.patch("dwarffi.dffi.get_dwarf2json_path", return_value="/usr/bin/dwarf2json"):
        with mock.patch("subprocess.run", side_effect=mock_subprocess_run):
            ffi.load_elf(
                str(fake_elf),
                save_isf_to=str(out_file),
            )

    # The ISF dictionary returned by our mock should be successfully loaded
    assert ffi.sizeof("custom_int") == 4

    # The resulting file should have been written and compressed
    assert out_file.exists()

    # Verify we can load the newly generated compressed file
    ffi_new = DFFI(str(out_file))
    assert ffi_new.sizeof("custom_int") == 4


def test_load_elf_missing_dwarf2json(tmp_path):
    ffi = DFFI()
    fake_elf = tmp_path / "test_binary.elf"
    fake_elf.write_bytes(b"\x7fELF...")

    # Simulate get_dwarf2json_path failing to find the executable
    with mock.patch("dwarffi.dffi.get_dwarf2json_path", return_value=None):
        with pytest.raises(RuntimeError, match="'dwarf2json' not found in PATH"):
            ffi.load_elf(str(fake_elf))


def test_load_elf_file_not_found():
    ffi = DFFI()
    
    with mock.patch("dwarffi.dffi.get_dwarf2json_path", return_value="/usr/bin/dwarf2json"):
        with pytest.raises(FileNotFoundError, match="Binary file not found"):
            ffi.load_elf("/this/does/not/exist.elf")


def test_load_elf_invalid_binary(tmp_path):
    ffi = DFFI()
    fake_elf = tmp_path / "bad_binary.elf"
    fake_elf.write_bytes(b"BAD DATA")

    def mock_subprocess_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=cmd,
            stderr="fatal: invalid ELF magic"
        )

    with mock.patch("dwarffi.dffi.get_dwarf2json_path", return_value="/usr/bin/dwarf2json"):
        with mock.patch("subprocess.run", side_effect=mock_subprocess_run):
            # Added (?s) so the .* will successfully jump over the \n in the error string
            with pytest.raises(RuntimeError, match=r"(?s)dwarf2json failed:.*invalid ELF magic"):
                ffi.load_elf(str(fake_elf))


def test_load_elf_invalid_json(tmp_path):
    ffi = DFFI()
    fake_elf = tmp_path / "test_binary.elf"
    fake_elf.write_bytes(b"\x7fELF...")

    def mock_subprocess_run(cmd, **kwargs):
        class MockCompletedProcess:
            def __init__(self, stdout):
                self.stdout = stdout
        return MockCompletedProcess("This is not valid JSON data")

    with mock.patch("dwarffi.dffi.get_dwarf2json_path", return_value="/usr/bin/dwarf2json"):
        with mock.patch("subprocess.run", side_effect=mock_subprocess_run):
            with pytest.raises(RuntimeError, match="Failed to parse dwarf2json output"):
                ffi.load_elf(str(fake_elf))


def test_load_elf_architectures(tmp_path):
    """Ensure the proper flags (--macho vs --elf) are passed based on the OS."""
    ffi = DFFI()
    fake_elf = tmp_path / "test_binary.elf"
    fake_elf.write_bytes(b"\x7fELF...")

    executed_cmds = []

    def mock_subprocess_run(cmd, **kwargs):
        class MockCompletedProcess:
            def __init__(self, stdout):
                self.stdout = stdout
        executed_cmds.append(cmd)
        
        fake_isf = {"base_types": {}, "user_types": {}, "symbols": {}, "enums": {}, "typedefs": {}}
        return MockCompletedProcess(json.dumps(fake_isf))

    # 1. Test Linux platform
    with mock.patch("dwarffi.dffi.get_dwarf2json_path", return_value="/usr/bin/dwarf2json"):
        with mock.patch("sys.platform", "linux"):
            with mock.patch("subprocess.run", side_effect=mock_subprocess_run):
                ffi.load_elf(str(fake_elf))
                
    assert executed_cmds[0] == ["/usr/bin/dwarf2json", "linux", "--elf", str(fake_elf)]
    executed_cmds.clear()

    # 2. Test Mac platform
    with mock.patch("dwarffi.dffi.get_dwarf2json_path", return_value="/usr/bin/dwarf2json"):
        with mock.patch("sys.platform", "darwin"):
            with mock.patch("subprocess.run", side_effect=mock_subprocess_run):
                ffi.load_elf(str(fake_elf))

    assert executed_cmds[0] == ["/usr/bin/dwarf2json", "mac", "--macho", str(fake_elf)]


def test_load_elf_bytes_success():
    """Test that the byte wrapper correctly manages the temp file lifecycle."""
    ffi = DFFI()
    fake_elf_data = b"\x7fELF..."

    # Mock the internal load_elf call so we don't have to stub subprocess again
    with mock.patch.object(ffi, "load_elf") as mock_load_elf:
        ffi.load_elf_bytes(fake_elf_data, dwarf2json_cmd="/custom/dwarf2json")
        
        mock_load_elf.assert_called_once()
        args, kwargs = mock_load_elf.call_args
        
        # Grab the temporary path it generated
        tmp_path = args[0]
        assert os.path.isabs(tmp_path)
        assert kwargs.get("dwarf2json_cmd") == "/custom/dwarf2json"
        
        # Verify the file was cleaned up upon exit
        assert not os.path.exists(tmp_path)

@pytest.mark.parametrize(
    "compiler, compiler_flags",
    [
        # Architectures with debug symbols
        ("gcc", ["-g", "-O0"]),
        ("clang", ["-g", "-O0", "-target", "x86_64-linux-gnu"]),
        ("arm-none-eabi-gcc", ["-g", "-mthumb", "-O1"]),
        ("aarch64-linux-gnu-gcc", ["-g", "-mabi=lp64"]),
        ("powerpc-linux-gnu-gcc", ["-g", "-m32"]),
        
        # Architectures without debug symbols (stripped / optimized)
        ("gcc", ["-O2", "-s"]),
        ("clang", ["-O3"]),
        ("arm-none-eabi-gcc", ["-O3", "-fomit-frame-pointer"]),
    ]
)
def test_cdef_hello_world_architectures(tmp_path, compiler, compiler_flags):
    """
    Tests compiling a simple hello world C program across various architectures 
    and verifies that dwarffi correctly handles binaries with and without debug symbols.
    """
    ffi = DFFI()
    source_code = """
    #include <stdio.h>
    int hello_world_value = 42;
    int main() {
        printf("Hello World\\n");
        return 0;
    }
    """
    
    # Determine if this parameter configuration includes debug symbols
    has_debug_symbols = "-g" in compiler_flags

    def mock_which(cmd, *args, **kwargs):
        if cmd == "dwarf2json":
            return "/usr/bin/dwarf2json"
        if cmd == compiler.split()[0]:
            return f"/usr/bin/{cmd}"
        return None

    def mock_subprocess_run(cmd, **kwargs):
        class MockCompletedProcess:
            def __init__(self, stdout):
                self.stdout = stdout

        cmd_str = " ".join(cmd)

        # 1. Simulate the C compiler succeeding
        if compiler in cmd_str:
            return MockCompletedProcess("")

        # 2. Simulate dwarf2json's reaction to the compiled object file
        if "dwarf2json" in cmd_str:
            if has_debug_symbols:
                # With -g, dwarf2json successfully extracts types and symbols
                fake_isf = {
                    "metadata": {},
                    "base_types": {
                        "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"}
                    },
                    "user_types": {},
                    "enums": {},
                    "symbols": {
                        "hello_world_value": {"address": 0x1000, "type_info": {"kind": "base", "name": "int"}},
                        "main": {"address": 0x1020, "type_info": {"kind": "function"}}
                    },
                    "typedefs": {},
                }
                return MockCompletedProcess(json.dumps(fake_isf))
            else:
                # Without -g, dwarf2json returns empty datasets (no DWARF info found)
                fake_empty_isf = {
                    "metadata": {},
                    "base_types": {},
                    "user_types": {},
                    "enums": {},
                    "symbols": {},
                    "typedefs": {},
                }
                return MockCompletedProcess(json.dumps(fake_empty_isf))

        raise ValueError(f"Unexpected command: {cmd}")

    with mock.patch("shutil.which", side_effect=mock_which):
        # Mock get_dwarf2json_path specifically if your dffi.py relies on it internally
        with mock.patch("dwarffi.dffi.get_dwarf2json_path", return_value="/usr/bin/dwarf2json"):
            with mock.patch("subprocess.run", side_effect=mock_subprocess_run):
                
                # Append -c so it creates an object file (as cdef does)
                ffi.cdef(
                    source=source_code,
                    compiler=compiler,
                    compiler_flags=compiler_flags + ["-c"]
                )
    
    # Assertions based on whether debug info was compiled in
    if has_debug_symbols:
        assert ffi.sizeof("int") == 4
        assert "hello_world_value" in ffi.symbols
        assert "main" in ffi.symbols
    else:
        # If no debug symbols were extracted, the types and symbols dicts should be empty
        with pytest.raises(KeyError, match="Unknown type 'int'"):
            ffi.sizeof("int")
        assert "hello_world_value" not in ffi.symbols
        assert "main" not in ffi.symbols