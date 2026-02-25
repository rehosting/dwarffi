import shutil
import struct
import pytest

from dwarffi import DFFI

# ---------------------------------------------------------------------------
# Setup for E2E Compiler Tests
# ---------------------------------------------------------------------------
COMPILERS = {
    "gcc": shutil.which("gcc"),
    "clang": shutil.which("clang"),
    "arm32": shutil.which("arm-linux-gnueabi-gcc"),
    "aarch64": shutil.which("aarch64-linux-gnu-gcc"),
}

AVAILABLE_COMPILERS = [path for name, path in COMPILERS.items() if path is not None]
HAS_DWARF2JSON = shutil.which("dwarf2json") is not None

pytestmark_e2e = pytest.mark.skipif(
    not HAS_DWARF2JSON or not AVAILABLE_COMPILERS, 
    reason="dwarf2json or compilers missing from PATH"
)

# ===========================================================================
# 1 & 4: Exotic Floats and SIMD Vectors (E2E)
# ===========================================================================

@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_exotic_floats_and_simd(compiler):
    """
    Tests handling of compiler-specific SIMD vector extensions and 
    exotic floats (like long double, which is often 80-bit / 16-bytes).
    """
    if not HAS_DWARF2JSON or not AVAILABLE_COMPILERS:
        pytest.skip("Missing compilers")
        
    ffi = DFFI()
    ffi.cdef(
        """
        // GCC/Clang generic vector extension (16 bytes / 128-bit)
        typedef float v4sf __attribute__ ((vector_size (16)));
        
        struct high_perf {
            long double exotic_float;
            v4sf vector_reg;
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct high_perf")
    
    # 1. Test the long double
    # The size of long double changes based on the CPU architecture!
    ld_size = ffi.sizeof("long double")
    exotic_val = inst.exotic_float
    
    if ld_size > 8:
        # On x86_64, it's 16 bytes. Python has no native 16-byte float,
        # so dwarffi safely falls back to giving us the raw bytes.
        assert isinstance(exotic_val, bytes)
        assert len(exotic_val) == ld_size
        
        test_payload = b'\xAA' * ld_size
        inst.exotic_float = test_payload
        assert inst.exotic_float == test_payload
    else:
        # On ARM32, it's 8 bytes. dwarffi natively maps this to a Python float!
        assert isinstance(exotic_val, float)
        
        inst.exotic_float = 3.14159
        assert abs(inst.exotic_float - 3.14159) < 0.001

    # 2. Test the SIMD vector
    # dwarf2json successfully recognizes the vector as an array of 4 floats.
    # dwarffi maps this into a seamless Python list-like view!
    assert len(inst.vector_reg) == 4
    
    # Write to specific lanes in the SIMD register
    inst.vector_reg[0] = 1.5
    inst.vector_reg[3] = 4.25
    
    assert inst.vector_reg[0] == 1.5
    assert inst.vector_reg[3] == 4.25


# ===========================================================================
# 2, 5, 6: C++ Quirks and Big-Endian Hardware (Mocked)
# ===========================================================================

def test_cpp_pointer_to_member():
    """
    In C++ (Itanium ABI), a pointer to a member function is 16 bytes, not 8!
    It contains a function pointer and an adjustment offset for `this`.
    """
    base_types = {
        "pointer": {"size": 16, "signed": False, "kind": "pointer", "endian": "little"},
    }
    ffi = DFFI({
        "metadata": {}, "base_types": base_types,
        "user_types": {
            "cpp_class": {
                "kind": "struct", "size": 32,
                "fields": {
                    "normal_int": {"offset": 0, "type": {"kind": "base", "name": "int", "size": 4}},
                    "member_func_ptr": {"offset": 8, "type": {"kind": "pointer", "subtype": {"name": "void"}}},
                }
            }
        },
        "enums": {}, "symbols": {}
    })
    
    buf = bytearray(32)
    inst = ffi.from_buffer("struct cpp_class", buf)
    
    # Writing a 16-byte address should work cleanly through the fallback engine
    huge_addr = 0x112233445566778899AABBCCDDEEFF
    inst.member_func_ptr = huge_addr
    
    # Ensure it didn't crash, and correctly read back all 16 bytes
    ptr = inst.member_func_ptr
    assert ptr.address == huge_addr


def test_cpp_hidden_vtable_pointers():
    """
    C++ classes with virtual functions have hidden vtable pointers injected
    by the compiler. DWARF usually gives these artificial names.
    """
    base_types = {
        "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"},
        "int": {"size": 4, "signed": True, "kind": "int", "endian": "little"},
    }
    ffi = DFFI({
        "metadata": {}, "base_types": base_types,
        "user_types": {
            "VirtualClass": {
                "kind": "struct", "size": 16,
                "fields": {
                    # The hidden vtable pointer emitted by DWARF
                    "_vptr$VirtualClass": {"offset": 0, "type": {"kind": "pointer", "subtype": {"name": "void"}}},
                    "user_data": {"offset": 8, "type": {"kind": "base", "name": "int"}},
                }
            }
        },
        "enums": {}, "symbols": {}
    })
    
    inst = ffi.from_buffer("struct VirtualClass", bytearray(16))
    
    inst.user_data = 42
    
    # The user can still access the hidden DWARF field if needed for deep introspection
    hidden_vtable = getattr(inst, "_vptr$VirtualClass")
    assert hidden_vtable.address == 0
    assert inst.user_data == 42


def test_big_endian_bitfield_packing():
    """
    Tests that Big-Endian memory layouts seamlessly work with our bitfield math.
    By parsing the bytes into a mathematical Python integer using `byteorder="big"`,
    standard bitwise shifts magically work without manual endian swapping!
    """
    base_types = {
        # Notice endian="big"
        "u16_be": {"size": 2, "signed": False, "kind": "int", "endian": "big"},
    }
    ffi = DFFI({
        "metadata": {}, "base_types": base_types,
        "user_types": {
            "network_header": {
                "kind": "struct", "size": 2,
                "fields": {
                    # Let's say DWARF tells us the target field is at bit offset 4, length 4
                    "flag": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 4, "bit_position": 4, "type": {"kind": "base", "name": "u16_be"}}},
                }
            }
        },
        "enums": {}, "symbols": {}
    })
    
    buf = bytearray([0xAB, 0xCD]) # Big endian memory for 0xABCD
    inst = ffi.from_buffer("struct network_header", buf)
    
    # The math integer is 0xABCD.
    # Shift right 4: 0x0ABC. Mask 4 bits: 0xC (12).
    assert inst.flag == 0xC
    
    # Write 0x9 (1001) to the bitfield
    inst.flag = 0x9
    
    # 0xABCD with bits 4-7 replaced by 0x9 becomes 0xAB9D.
    # Because it's big endian, memory should now be [0xAB, 0x9D].
    assert buf[0] == 0xAB
    assert buf[1] == 0x9D