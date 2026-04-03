import pytest

from dwarffi.dffi import DFFI
from dwarffi.instances import BoundArrayView, BoundTypeInstance, Ptr

# ISF with a variety of types to test access logic
TEST_ISF = {
    "metadata": {"format": "1.4.0"},
    "base_types": {
        "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
        "char": {"kind": "char", "size": 1, "signed": True, "endian": "little"},
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
        },
        "Container": {
            "kind": "struct",
            "size": 24,
            "fields": {
                "p_arr": {"type": {"kind": "array", "count": 2, "subtype": {"kind": "struct", "name": "Point"}}, "offset": 0},
                "i_arr": {"type": {"kind": "array", "count": 2, "subtype": {"kind": "base", "name": "int"}}, "offset": 16},
            }
        }
    },
    "enums": {
        "Status": {
            "size": 4,
            "constants": {"OK": 0, "ERROR": 1},
        }
    },
    "symbols": {},
    "typedefs": {},
}

@pytest.fixture
def d():
    return DFFI(isf_input=TEST_ISF)

def test_array_access_types(d: DFFI):
    """Verify that array element access returns correctly unboxed/boxed types."""
    
    # 1. Base Type Array -> returns raw Python values (ints)
    int_arr_t = d.t.int.array(2)
    int_arr = int_arr_t([10, 20])
    assert isinstance(int_arr[0], int)
    assert int_arr[0] == 10

    # 2. Enum Array -> returns EnumInstance (value-like with .name)
    enum_arr_t = d.t.Status.array(2)
    enum_arr = enum_arr_t([0, 1])
    # Should be our fancy EnumInstance, not a BoundTypeInstance
    assert type(enum_arr[0]).__name__ == "EnumInstance"
    assert enum_arr[0].name == "OK"
    assert int(enum_arr[1]) == 1

    # 3. Struct Array -> returns BoundTypeInstance (boxed container)
    point_arr_t = d.t.Point.array(2)
    point_arr = point_arr_t([{"x": 1, "y": 2}, {"x": 3, "y": 4}])
    assert isinstance(point_arr[0], BoundTypeInstance)
    assert point_arr[0].x == 1
    # Check that it's still bound to the array's buffer
    point_arr[0].x = 99
    assert point_arr[0].x == 99

    # 4. Pointer Array -> returns Ptr objects
    ptr_arr_t = d.t.int.ptr.array(2)
    ptr_arr = ptr_arr_t([0x1000, 0x2000])
    assert isinstance(ptr_arr[0], Ptr)
    assert ptr_arr[0].address == 0x1000


def test_struct_field_access_types(d: DFFI):
    """Verify that struct field access follows the same unboxing rules."""
    
    # Instantiate a Container
    c = d.t.Container(
        p_arr=[{"x": 10, "y": 20}, {"x": 30, "y": 40}],
        i_arr=[50, 60]
    )

    # Accessing an array field should return a BoundArrayView
    assert isinstance(c.p_arr, BoundArrayView)
    
    # Accessing an element of that array field (Struct)
    assert isinstance(c.p_arr[0], BoundTypeInstance)
    assert c.p_arr[0].y == 20

    # Accessing an element of that array field (Base Type)
    assert isinstance(c.i_arr[0], int)
    assert c.i_arr[1] == 60


def test_nested_derived_call_types(d: DFFI):
    """Verify that calling derived types (sugar) returns the correct objects."""
    
    # Calling an array type directly returns the BoundArrayView
    arr_instance = d.t.int.array(5)([1, 2, 3, 4, 5])
    assert isinstance(arr_instance, BoundArrayView)
    assert arr_instance[2] == 3

    # Calling a pointer type directly returns a Ptr object (unboxed)
    ptr_instance = d.t.Point.ptr(0xDEADBEEF)
    assert isinstance(ptr_instance, Ptr)
    assert ptr_instance.address == 0xDEADBEEF
    assert ptr_instance.points_to_type_info["name"] == "Point"