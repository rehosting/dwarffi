import os
import shutil
import pytest

from dwarffi import DFFI

COMPILERS = {
    "gcc": shutil.which("gcc"),
    "clang": shutil.which("clang"),
    "arm32": shutil.which("arm-linux-gnueabi-gcc"),
    "aarch64": shutil.which("aarch64-linux-gnu-gcc"),
}

AVAILABLE_COMPILERS = [path for name, path in COMPILERS.items() if path is not None]
HAS_DWARF2JSON = shutil.which("dwarf2json") is not None

pytestmark = pytest.mark.skipif(
    not HAS_DWARF2JSON or not AVAILABLE_COMPILERS, 
    reason="dwarf2json or compilers missing from PATH"
)


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_floating_point_and_bools(compiler):
    """Tests that exotic and standard floats/bools parse correctly via the struct module."""
    ffi = DFFI()
    ffi.cdef(
        """
        #include <stdbool.h>
        
        struct physics_data {
            bool is_active;
            float velocity;
            double mass;
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct physics_data")
    inst.is_active = True
    inst.velocity = 9.81
    inst.mass = 3.14159265359
    
    assert inst.is_active is True
    # Floating point comparisons need error margins
    assert abs(inst.velocity - 9.81) < 0.001
    assert abs(inst.mass - 3.14159265359) < 0.000000001


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_stdint_types(compiler):
    """Tests strict width enforcement and sign boundaries matching stdint.h definitions."""
    ffi = DFFI()
    ffi.cdef(
        """
        #include <stdint.h>
        
        struct stdint_data {
            int8_t i8;
            uint8_t u8;
            int16_t i16;
            uint16_t u16;
            int32_t i32;
            uint32_t u32;
            int64_t i64;
            uint64_t u64;
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct stdint_data")
    
    # Pack them to their absolute limits
    inst.i8 = -128
    inst.u8 = 255
    
    inst.i16 = -32768
    inst.u16 = 65535
    
    inst.i32 = -2147483648
    inst.u32 = 4294967295
    
    inst.i64 = -9223372036854775808
    inst.u64 = 18446744073709551615
    
    # Ensure they read out correctly without overflow/underflow
    assert inst.i8 == -128
    assert inst.u8 == 255
    assert inst.i16 == -32768
    assert inst.u16 == 65535
    assert inst.i32 == -2147483648
    assert inst.u32 == 4294967295
    assert inst.i64 == -9223372036854775808
    assert inst.u64 == 18446744073709551615