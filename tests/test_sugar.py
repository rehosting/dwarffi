import pytest

from dwarffi.dffi import DFFI

TEST_ISF = {
    "metadata": {"format": "1.4.0"},
    "base_types": {
        "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
        "long": {"kind": "int", "size": 8, "signed": True, "endian": "little"},
    },
    "user_types": {
        "Point": {
            "kind": "struct",
            "size": 8,
            "fields": {
                "x": {"type": {"kind": "base", "name": "int"}, "offset": 0},
                "y": {"type": {"kind": "base", "name": "int"}, "offset": 4},
            },
        },
    },
    "enums": {},
    "symbols": {
        "global_point": {"address": 0x4000, "type": {"kind": "struct", "name": "Point"}}
    },
    "typedefs": {},
}

@pytest.fixture
def d():
    """Provides a DFFI instance loaded with the test ISF."""
    return DFFI(isf_input=TEST_ISF)

def test_namespace_sugar(d: DFFI):
    """Test Feature 1: d.t and d.sym namespaces"""
    # 1. Type resolution
    t_int = d.t.int
    assert t_int.size == 4
    
    t_point = d.t.Point
    assert t_point.size == 8

    # 2. Symbol resolution
    sym = d.sym.global_point
    assert sym.address == 0x4000

    # 3. Error handling
    with pytest.raises(AttributeError, match="Type 'invalid' not found"):
        _ = d.t.invalid
        
    with pytest.raises(AttributeError, match="Symbol 'missing' not found"):
        _ = d.sym.missing

def test_type_modifiers_sugar(d: DFFI):
    """Test Feature 2: .ptr and .array() on type objects"""
    int_t = d.t.int
    
    # Pointer generation
    int_ptr = int_t.ptr
    assert isinstance(int_ptr, dict)
    assert int_ptr["kind"] == "pointer"
    assert int_ptr["subtype"]["name"] == "int"
    
    # Array generation
    int_arr = int_t.array(5)
    assert isinstance(int_arr, dict)
    assert int_arr["kind"] == "array"
    assert int_arr["count"] == 5
    assert int_arr["subtype"]["name"] == "int"
    
    # Verify the generated array dictionary works directly with d.new()
    arr_instance = d.new(int_arr, [10, 20, 30, 40, 50])
    assert len(arr_instance) == 5
    assert arr_instance[2] == 30

def test_pointer_arithmetic(d: DFFI):
    """Test Feature 3: __add__ and __sub__ on Ptr instances"""
    # Create an int pointer starting at 0x1000
    int_ptr_t = d.t.int.ptr
    p1 = d.cast(int_ptr_t, 0x1000)
    
    # sizeof(int) is 4, so +2 should add 8 bytes
    p2 = p1 + 2
    assert p2.address == 0x1008
    assert p2.points_to_type_info["name"] == "int"
    
    # Subtract 1 should remove 4 bytes
    p3 = p2 - 1
    assert p3.address == 0x1004
    
    # Test with a larger struct type (Point is 8 bytes)
    point_ptr_t = d.t.Point.ptr
    p_struct = d.cast(point_ptr_t, 0x2000)
    
    # +3 should add 24 bytes (3 * 8)
    p_struct_advanced = p_struct + 3
    assert p_struct_advanced.address == 0x2018
    
    # Ensure invalid additions fallback gracefully
    with pytest.raises(TypeError):
        _ = p1 + "string"