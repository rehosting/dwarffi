import pytest

from dwarffi.backend import BytesBackend
from dwarffi.dffi import DFFI

# A minimal ISF covering primitives, nested structs, and arrays
DUMMY_ISF = {
    "metadata": {},
    "base_types": {
        "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
    },
    "user_types": {
        "Inner": {
            "kind": "struct", "size": 4,
            "fields": {
                "val": {"offset": 0, "type": {"kind": "base", "name": "int"}}
            }
        },
        "Outer": {
            "kind": "struct", "size": 24,
            "fields": {
                "inner": {"offset": 0, "type": {"kind": "struct", "name": "Inner"}},
                "arr": {"offset": 4, "type": {"kind": "array", "count": 5, "subtype": {"kind": "base", "name": "int"}}}
            }
        }
    },
    "enums": {}, "symbols": {}
}

@pytest.fixture
def ffi():
    # Use a dummy memory backend so we can safely test pointer dereferencing
    return DFFI(DUMMY_ISF, backend=BytesBackend(bytearray(64)))

def test_addressof_normalizes_vtype_to_dict(ffi):
    """
    Ensures that taking the address of a whole struct converts the Vtype object
    into a standard ISF dictionary to prevent crashes in Ptr.deref().
    """
    inst = ffi.new("struct Inner")
    ptr = ffi.addressof(inst)
    
    # Verify the type info was normalized
    assert isinstance(ptr.points_to_type_info, dict)
    assert ptr.points_to_type_info["kind"] == "struct"
    assert ptr.points_to_type_info["name"] == "Inner"

def test_addressof_nested_field(ffi):
    """
    Ensures taking the address of a nested struct field computes the correct
    offset and maintains the correct type context.
    """
    inst = ffi.new("struct Outer")
    ptr = ffi.addressof(inst, "inner")
    
    assert isinstance(ptr.points_to_type_info, dict)
    assert ptr.points_to_type_info["kind"] == "struct"
    assert ptr.points_to_type_info["name"] == "Inner"
    
    # "inner" is at offset 0
    assert ptr.address == inst._instance_offset + 0

def test_addressof_array_decay(ffi):
    """
    Ensures that taking the address of an array field perfectly mimics C's
    array-to-pointer decay, returning a pointer to the array's elements.
    """
    inst = ffi.new("struct Outer")
    ptr = ffi.addressof(inst, "arr")
    
    # "arr" is an array of ints, so the pointer should decay to an int pointer
    assert isinstance(ptr.points_to_type_info, dict)
    assert ptr.points_to_type_info["kind"] == "base"
    assert ptr.points_to_type_info["name"] == "int"
    
    # "arr" starts at offset 4
    assert ptr.address == inst._instance_offset + 4

def test_addressof_deref_safety(ffi):
    """
    Proves that the dictionary normalization allows the resulting pointer
    to be safely dereferenced without throwing an AttributeError inside the 
    _resolve_type_info engine.
    """
    # Create an instance backed by absolute memory
    inst = ffi.from_address("struct Inner", 16)
    
    # Take its address
    ptr = ffi.addressof(inst)
    
    # This deref() call would previously crash due to the Ptr class receiving 
    # a VtypeUserType object instead of a TypeInfoDict.
    derefed_inst = ptr.deref()
    
    # Verify the dereferenced object works
    assert derefed_inst._instance_type_name == "Inner"
    assert derefed_inst._instance_offset == 16