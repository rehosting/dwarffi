import pytest
from dwarffi.dffi import DFFI
from dwarffi.types import VtypeDerived

# A minimal ISF for testing without needing external compilers
TEST_ISF = {
    "metadata": {"format": "1.4.0"},
    "base_types": {
        "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
        # A base 'pointer' type is required by DFFI to know pointer sizes
        "pointer": {"kind": "int", "size": 8, "signed": False, "endian": "little"},
    },
    "user_types": {
        "Point": {
            "kind": "struct",
            "size": 8,
            "fields": {
                "x": {"type": {"kind": "base", "name": "int"}, "offset": 0},
                "y": {"type": {"kind": "base", "name": "int"}, "offset": 4},
            },
        }
    },
    "enums": {
        "Color": {
            "size": 4,
            "constants": {"RED": 0, "GREEN": 1, "BLUE": 2},
        }
    },
    "symbols": {},
    "typedefs": {},
}


@pytest.fixture
def d():
    """Provides a DFFI instance loaded with the test ISF."""
    return DFFI(isf_input=TEST_ISF)


def test_derived_array_sugar(d: DFFI):
    """Test that .array() returns a VtypeDerived that can be called."""
    int_t = d.t.int
    arr_t = int_t.array(3)
    
    # Verify it acts exactly like the ISF dictionary
    assert isinstance(arr_t, dict)
    assert isinstance(arr_t, VtypeDerived)
    assert arr_t["kind"] == "array"
    assert arr_t["count"] == 3
    
    # Instantiate it directly using the sugar
    arr = arr_t([10, 20, 30])
    assert len(arr) == 3
    assert arr[0] == 10
    assert arr[2] == 30


def test_derived_pointer_sugar(d: DFFI):
    """Test that .ptr returns a VtypeDerived that can be called."""
    point_t = d.t.Point
    ptr_t = point_t.ptr
    
    # Verify inheritance and dictionary access
    assert isinstance(ptr_t, VtypeDerived)
    assert ptr_t["kind"] == "pointer"
    assert ptr_t["subtype"]["name"] == "Point"
    
    # Instantiate the pointer directly at address 0x4000
    ptr = ptr_t(0x4000)
    assert ptr.address == 0x4000
    assert ptr.points_to_type_info["name"] == "Point"


def test_derived_chaining(d: DFFI):
    """Test that .ptr and .array() can be chained endlessly."""
    int_t = d.t.int
    
    # Create an array of pointers (int *arr[5])
    arr_of_ptr_t = int_t.ptr.array(5)
    assert isinstance(arr_of_ptr_t, VtypeDerived)
    assert arr_of_ptr_t["kind"] == "array"
    assert arr_of_ptr_t["subtype"]["kind"] == "pointer"


def test_derived_enum_array(d: DFFI):
    """Test arrays of enums using the syntactic sugar."""
    color_t = d.t.Color
    color_arr_t = color_t.array(2)
    
    arr = color_arr_t([1, 2])
    # DFFI should correctly resolve the enum strings
    assert arr[0].name == "GREEN"
    assert arr[1].name == "BLUE"


def test_derived_unbound_error():
    """Test that an unbound VtypeDerived raises a helpful error."""
    # Create a raw derived type not tied to an engine
    raw_arr = VtypeDerived({"kind": "array", "count": 2, "subtype": {"name": "int"}})
    
    with pytest.raises(RuntimeError, match="Derived type is not bound to a DFFI engine"):
        raw_arr([1, 2])


def test_ctype_string_parser_returns_derived_types(d: DFFI):
    """Test that parsing C-strings natively returns VtypeDerived objects."""
    # Parse an array
    arr_t = d.typeof("int[4]")
    assert isinstance(arr_t, VtypeDerived)
    
    arr = arr_t([100, 200, 300, 400])
    assert len(arr) == 4
    assert arr[3] == 400
    
    # Parse a pointer
    ptr_t = d.typeof("Point *")
    assert isinstance(ptr_t, VtypeDerived)

    ptr = ptr_t(0x8000)
    assert ptr.address == 0x8000
    assert ptr.points_to_type_info["name"] == "Point"