import json
import lzma
import os
import shutil
import subprocess

import pytest

from dwarffi import DFFI

# Check for the primary dependency
HAS_DWARF2JSON = shutil.which("dwarf2json") is not None

# Check for various compilers
GCC = shutil.which("gcc")
CLANG = shutil.which("clang")
ARM32 = shutil.which("arm-linux-gnueabi-gcc")
AARCH64 = shutil.which("aarch64-linux-gnu-gcc")

pytestmark = pytest.mark.skipif(
    not HAS_DWARF2JSON, reason="dwarf2json not found in PATH. Integration tests skipped."
)


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_native_gcc():
    ffi = DFFI()
    ffi.cdef(
        """
        struct native_test {
            int a;
            long b;
            void* p;
        };
        enum my_state { STATE_IDLE = 0, STATE_RUNNING = 1 };
    """,
        compiler=GCC,
    )

    # Verify the types were extracted
    assert ffi.sizeof("struct native_test") > 0
    assert ffi.sizeof("enum my_state") > 0

    # Ensure we can instantiate and use them
    inst = ffi.new("struct native_test", {"a": 42})
    assert inst.a == 42


@pytest.mark.skipif(CLANG is None, reason="clang not found in PATH")
def test_cdef_native_clang():
    ffi = DFFI()
    ffi.cdef(
        """
        typedef unsigned char uint8_t;
        struct clang_test {
            uint8_t flags;
        };
    """,
        compiler=CLANG,
    )

    assert ffi.sizeof("struct clang_test") == 1
    inst = ffi.new("struct clang_test", {"flags": 255})
    assert inst.flags == 255


@pytest.mark.skipif(ARM32 is None, reason="arm-linux-gnueabi-gcc not found in PATH")
def test_cdef_cross_compile_arm32():
    """Tests that cross-compiling properly reflects 32-bit architecture sizes."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct ptr_struct {
            void* ptr1;
            void* ptr2;
        };
    """,
        compiler=ARM32,
    )

    # 32-bit ARM has 4-byte pointers, so 2 pointers = 8 bytes
    assert ffi.sizeof("struct ptr_struct") == 8
    assert ffi.sizeof("void *") == 4


@pytest.mark.skipif(AARCH64 is None, reason="aarch64-linux-gnu-gcc not found in PATH")
def test_cdef_cross_compile_aarch64():
    """Tests that cross-compiling properly reflects 64-bit architecture sizes."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct ptr_struct {
            void* ptr1;
            void* ptr2;
        };
    """,
        compiler=AARCH64,
    )

    # 64-bit ARM has 8-byte pointers, so 2 pointers = 16 bytes
    assert ffi.sizeof("struct ptr_struct") == 16
    assert ffi.sizeof("void *") == 8


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_compiler_error():
    """Ensures invalid C code properly raises a RuntimeError with compiler output."""
    ffi = DFFI()
    with pytest.raises(RuntimeError, match="Compilation failed"):
        ffi.cdef(
            """
            struct bad_struct {
                unknown_type a; // This will fail compilation
            };
        """,
            compiler=GCC,
        )


# ----------------------------
# save_isf_to behavior
# ----------------------------


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_save_isf_to_json_writes_valid_json(tmp_path):
    ffi = DFFI()
    out = tmp_path / "out.json"

    ffi.cdef(
        """
        struct native_test { int a; long b; void* p; };
        enum my_state { STATE_IDLE = 0, STATE_RUNNING = 1 };
        """,
        compiler=GCC,
        save_isf_to=str(out),
    )

    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))

    # Smoke-check ISF structure (don’t overfit to exact schema versions)
    assert "base_types" in data
    assert "symbols" in data
    assert "user_types" in data
    assert "enums" in data

    # Also verify the types actually loaded into the FFI
    assert ffi.sizeof("struct native_test") > 0
    assert ffi.sizeof("enum my_state") > 0


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_save_isf_to_json_xz_writes_valid_json(tmp_path):
    ffi = DFFI()
    out = tmp_path / "out.json.xz"

    ffi.cdef(
        """
        struct s { int a; };
        enum e { E0 = 0, E1 = 1 };
        """,
        compiler=GCC,
        save_isf_to=str(out),
    )

    assert out.exists()
    with lzma.open(out, "rt", encoding="utf-8") as f:
        data = json.load(f)

    assert "base_types" in data
    assert "user_types" in data
    assert "enums" in data

    assert ffi.sizeof("struct s") == 4
    assert ffi.sizeof("enum e") > 0


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_save_isf_to_creates_parent_dirs(tmp_path):
    ffi = DFFI()
    out_dir = tmp_path / "a" / "b" / "c"
    out = out_dir / "out.json"

    assert not out_dir.exists()

    ffi.cdef(
        "struct t { int x; };",
        compiler=GCC,
        save_isf_to=str(out),
    )

    assert out_dir.exists()
    assert out.exists()


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_save_isf_to_rejects_bad_extension(tmp_path):
    ffi = DFFI()
    out = tmp_path / "out.txt"

    with pytest.raises(ValueError, match=r"must end with '\.json' or '\.json\.xz'"):
        ffi.cdef("struct t { int x; };", compiler=GCC, save_isf_to=str(out))


# ----------------------------
# Multi-call / ordering behavior
# ----------------------------


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_multiple_calls_accumulate_types(tmp_path):
    ffi = DFFI()

    ffi.cdef("struct a { int x; };", compiler=GCC)
    ffi.cdef("struct b { int y; };", compiler=GCC)

    assert ffi.sizeof("struct a") == 4
    assert ffi.sizeof("struct b") == 4

    # Verify file order grew
    assert len(ffi._file_order) >= 2


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_same_type_name_first_wins_semantics():
    """
    Current behavior: if multiple cdef() calls define the same type name,
    resolution uses the earliest definition (first wins).
    """
    ffi = DFFI()

    ffi.cdef("struct dup { int a; };", compiler=GCC)
    assert ffi.sizeof("struct dup") == 4

    ffi.cdef("struct dup { long a; };", compiler=GCC)

    # first definition is still the one resolved
    assert ffi.sizeof("struct dup") == 4

    inst = ffi.new("struct dup", {"a": 7})
    assert inst.a == 7


# ----------------------------
# Compiler flags & debug retention sanity
# ----------------------------


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_retains_unused_debug_types_by_default():
    """
    Validates your default flags prevent pruning of DWARF type DIEs.
    These types are otherwise 'unused' from a linker perspective.
    """
    ffi = DFFI()
    ffi.cdef(
        """
        struct prune_me { int a; };
        enum prune_enum { P0 = 0, P1 = 1 };
        """,
        compiler=GCC,
    )

    assert ffi.sizeof("struct prune_me") == 4
    assert ffi.sizeof("enum prune_enum") > 0


# ----------------------------
# Error handling / diagnostics
# ----------------------------


def test_cdef_missing_compiler_raises():
    ffi = DFFI()
    with pytest.raises(RuntimeError, match=r"Compiler 'definitely-not-a-compiler' not found"):
        ffi.cdef("struct t { int x; };", compiler="definitely-not-a-compiler")


def test_cdef_missing_dwarf2json_raises(monkeypatch):
    ffi = DFFI()

    real_which = shutil.which

    def fake_which(name):
        if name == "dwarf2json":
            return None
        return real_which(name)

    monkeypatch.setattr(shutil, "which", fake_which)

    # Match the actual error prefix (quotes + PATH.)
    with pytest.raises(RuntimeError, match=r"'dwarf2json' not found in PATH\."):
        ffi.cdef("struct t { int x; };", compiler="gcc")


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_compiler_error_contains_command_and_stderr():
    ffi = DFFI()
    with pytest.raises(RuntimeError) as ei:
        ffi.cdef(
            """
            struct bad_struct {
                unknown_type a;
            };
            """,
            compiler=GCC,
        )

    msg = str(ei.value)
    assert "Compilation failed" in msg
    # Useful for debugging: include the compile command and stderr
    assert "Command:" in msg
    assert "Stderr:" in msg


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_dwarf2json_failure_raises_runtimeerror(monkeypatch, tmp_path):
    """
    Force dwarf2json invocation failure (simulate nonzero exit).
    """
    ffi = DFFI()

    def fake_run(*args, **kwargs):
        cmd = args[0]
        # Let compilation succeed; fail on dwarf2json
        if isinstance(cmd, list) and cmd and os.path.basename(cmd[0]) == "dwarf2json":
            raise subprocess.CalledProcessError(returncode=2, cmd=cmd, output="", stderr="boom")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    import subprocess as _subprocess

    monkeypatch.setattr(_subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match=r"dwarf2json failed"):
        ffi.cdef("struct t { int x; };", compiler=GCC)


# ----------------------------
# Architecture / pointer width tests
# ----------------------------


@pytest.mark.skipif(ARM32 is None, reason="arm-linux-gnueabi-gcc not found in PATH")
def test_cdef_cross_compile_arm32_pointer_size():
    ffi = DFFI()
    ffi.cdef(
        """
        struct ptr_struct {
            void* ptr1;
            void* ptr2;
        };
        """,
        compiler=ARM32,
    )
    assert ffi.sizeof("void *") == 4
    assert ffi.sizeof("struct ptr_struct") == 8


@pytest.mark.skipif(AARCH64 is None, reason="aarch64-linux-gnu-gcc not found in PATH")
def test_cdef_cross_compile_aarch64_pointer_size():
    ffi = DFFI()
    ffi.cdef(
        """
        struct ptr_struct {
            void* ptr1;
            void* ptr2;
        };
        """,
        compiler=AARCH64,
    )
    assert ffi.sizeof("void *") == 8
    assert ffi.sizeof("struct ptr_struct") == 16


# ----------------------------
# Clang-specific sanity
# ----------------------------


@pytest.mark.skipif(CLANG is None, reason="clang not found in PATH")
def test_cdef_clang_typedef_and_struct_layout():
    ffi = DFFI()
    ffi.cdef(
        """
        typedef unsigned char uint8_t;
        struct clang_test { uint8_t flags; };
        """,
        compiler=CLANG,
    )
    assert ffi.sizeof("struct clang_test") == 1
    inst = ffi.new("struct clang_test", {"flags": 255})
    assert inst.flags == 255
