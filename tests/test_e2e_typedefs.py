import pytest
from dwarffi.dffi import DFFI
from dwarffi.instances import BoundTypeInstance, Ptr, BoundArrayView

def test_e2e_new_dwarf2json_typedefs():
    """
    Comprehensive E2E test for typedef resolution using the updated dwarf2json.
    Verifies that standard typedefs are preserved in the ISF and resolvable by dwarffi.
    """
    d = DFFI()
    
    # 1. Compile C code packed with various typedef patterns
    d.cdef("""
        typedef unsigned char asdf8;
        typedef unsigned int u32;
        typedef u32 uint;               // Chained typedef
        typedef uint handle_t;          // Double chained typedef
        
        // Typedef for an anonymous struct
        typedef struct {
            asdf8 version;
            u32 id;
        } header_t;

        typedef int* int_ptr;           // Pointer typedef
        typedef float matrix_t[4][4];   // Multidimensional array typedef
    """)

    # 2. Verify the ISF backend actually received the typedefs from the new dwarf2json
    primary_isf_path = d._file_order[0]
    typedefs_dict = d.vtypejsons[primary_isf_path]._isf.typedefs
    
    assert "asdf8" in typedefs_dict
    assert "u32" in typedefs_dict
    assert "uint" in typedefs_dict
    assert "handle_t" in typedefs_dict
    assert "int_ptr" in typedefs_dict
    assert "matrix_t" in typedefs_dict

    # 3. Test Base Typedefs (should resolve to base ints/chars and return unboxed values)
    val_u8 = d.t.asdf8(255)
    assert val_u8 == 255
    assert isinstance(val_u8, int)
    assert d.sizeof("asdf8") == 1

    # 4. Test Chained Typedefs (handle_t -> uint -> u32 -> unsigned int)
    val_handle = d.t.handle_t(0xDEADBEEF)
    assert val_handle == 0xDEADBEEF
    assert d.sizeof("handle_t") == 4

    # 5. Test Struct Typedefs (should return BoundTypeInstance)
    header = d.t.header_t(version=1, id=123)
    assert isinstance(header, BoundTypeInstance)
    assert header.version == 1
    assert header.id == 123
    assert d.sizeof("header_t") >= 5  # At least 5, likely 8 due to padding

    # 6. Test Pointer Typedefs (should resolve to pointer dictionary and return Ptr)
    ptr = d.t.int_ptr(0x4000)
    assert isinstance(ptr, Ptr)
    assert ptr.address == 0x4000
    
    # Verify the pointer knows what it points to
    assert ptr.points_to_type_info.get("name") == "int"

    # 7. Test Array Typedefs (should resolve to array dictionary and return BoundArrayView)
    matrix = d.t.matrix_t()
    assert isinstance(matrix, BoundArrayView)
    assert len(matrix) == 4
    assert len(matrix[0]) == 4
    assert d.sizeof("matrix_t") == 4 * 4 * 4  # 4 rows * 4 cols * 4 bytes (float)