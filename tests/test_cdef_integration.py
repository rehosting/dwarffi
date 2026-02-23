import shutil
import pytest
from dwarffi.dffi import DFFI

# Check for the primary dependency
HAS_DWARF2JSON = shutil.which("dwarf2json") is not None

# Check for various compilers
GCC = shutil.which("gcc")
CLANG = shutil.which("clang")
ARM32 = shutil.which("arm-linux-gnueabi-gcc")
AARCH64 = shutil.which("aarch64-linux-gnu-gcc")

pytestmark = pytest.mark.skipif(
    not HAS_DWARF2JSON, 
    reason="dwarf2json not found in PATH. Integration tests skipped."
)


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_native_gcc():
    ffi = DFFI()
    ffi.cdef("""
        struct native_test {
            int a;
            long b;
            void* p;
        };
        enum my_state { STATE_IDLE = 0, STATE_RUNNING = 1 };
    """, compiler=GCC)
    
    # Verify the types were extracted
    assert ffi.sizeof("struct native_test") > 0
    assert ffi.sizeof("enum my_state") > 0
    
    # Ensure we can instantiate and use them
    inst = ffi.new("struct native_test", {"a": 42})
    assert inst.a == 42


@pytest.mark.skipif(CLANG is None, reason="clang not found in PATH")
def test_cdef_native_clang():
    ffi = DFFI()
    ffi.cdef("""
        typedef unsigned char uint8_t;
        struct clang_test {
            uint8_t flags;
        };
    """, compiler=CLANG)
    
    assert ffi.sizeof("struct clang_test") == 1
    inst = ffi.new("struct clang_test", {"flags": 255})
    assert inst.flags == 255


@pytest.mark.skipif(ARM32 is None, reason="arm-linux-gnueabi-gcc not found in PATH")
def test_cdef_cross_compile_arm32():
    """Tests that cross-compiling properly reflects 32-bit architecture sizes."""
    ffi = DFFI()
    ffi.cdef("""
        struct ptr_struct {
            void* ptr1;
            void* ptr2;
        };
    """, compiler=ARM32)
    
    # 32-bit ARM has 4-byte pointers, so 2 pointers = 8 bytes
    assert ffi.sizeof("struct ptr_struct") == 8
    assert ffi.sizeof("void *") == 4


@pytest.mark.skipif(AARCH64 is None, reason="aarch64-linux-gnu-gcc not found in PATH")
def test_cdef_cross_compile_aarch64():
    """Tests that cross-compiling properly reflects 64-bit architecture sizes."""
    ffi = DFFI()
    ffi.cdef("""
        struct ptr_struct {
            void* ptr1;
            void* ptr2;
        };
    """, compiler=AARCH64)
    
    # 64-bit ARM has 8-byte pointers, so 2 pointers = 16 bytes
    assert ffi.sizeof("struct ptr_struct") == 16
    assert ffi.sizeof("void *") == 8


@pytest.mark.skipif(GCC is None, reason="gcc not found in PATH")
def test_cdef_compiler_error():
    """Ensures invalid C code properly raises a RuntimeError with compiler output."""
    ffi = DFFI()
    with pytest.raises(RuntimeError, match="Compilation failed"):
        ffi.cdef("""
            struct bad_struct {
                unknown_type a; // This will fail compilation
            };
        """, compiler=GCC)