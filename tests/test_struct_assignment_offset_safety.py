import pytest
from dwarffi import DFFI

@pytest.fixture
def assignment_ffi():
    """Provides a DFFI instance with various structs and unions for testing."""
    isf = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "short": {"kind": "int", "size": 2, "signed": True, "endian": "little"},
        },
        "user_types": {
            "Point2D": {
                "kind": "struct", "size": 8,
                "fields": {
                    "x": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "y": {"offset": 4, "type": {"kind": "base", "name": "int"}}
                }
            },
            "Point3D": {
                "kind": "struct", "size": 12,
                "fields": {
                    "x": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "y": {"offset": 4, "type": {"kind": "base", "name": "int"}},
                    "z": {"offset": 8, "type": {"kind": "base", "name": "int"}}
                }
            },
            "BoundingBox": {
                "kind": "struct", "size": 16,
                "fields": {
                    "top_left": {"offset": 0, "type": {"kind": "struct", "name": "Point2D"}},
                    "bottom_right": {"offset": 8, "type": {"kind": "struct", "name": "Point2D"}}
                }
            },
            "PaddedStruct": {
                "kind": "struct", "size": 16,
                "fields": {
                    "before": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "pt": {"offset": 4, "type": {"kind": "struct", "name": "Point2D"}},
                    "after": {"offset": 12, "type": {"kind": "base", "name": "int"}}
                }
            },
            "VectorUnion": {
                "kind": "union", "size": 12,
                "fields": {
                    "p2": {"offset": 0, "type": {"kind": "struct", "name": "Point2D"}},
                    "p3": {"offset": 0, "type": {"kind": "struct", "name": "Point3D"}}
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    return DFFI(isf)


def test_struct_assignment_offset_safety(assignment_ffi):
    """
    Ensures that assigning a struct to a field in the middle of another struct
    does not overwrite the fields before or after it.
    """
    ffi = assignment_ffi
    padded = ffi.new("struct PaddedStruct")
    padded.before = 0x7AAAAAAA
    padded.after = 0x7BBBBBBB

    pt = ffi.new("struct Point2D")
    pt.x = 100
    pt.y = 200

    # Assign to the middle struct
    padded.pt = pt

    # Verify the struct fields were copied
    assert padded.pt.x == 100
    assert padded.pt.y == 200

    # Verify bounds weren't overwritten
    assert padded.before == 0x7AAAAAAA
    assert padded.after == 0x7BBBBBBB


def test_struct_nested_assignment_and_cache(assignment_ffi):
    """
    Tests assigning a struct to a nested field and verifies that DFFI's
    internal caching mechanism properly invalidates stale field data.
    """
    ffi = assignment_ffi
    box = ffi.new("struct BoundingBox")
    
    # Pre-populate and cache the read
    box.top_left.x = 5
    box.top_left.y = 10
    _ = box.top_left.x  # Triggers the field to cache

    # Create new point
    new_pt = ffi.new("struct Point2D")
    new_pt.x = 50
    new_pt.y = 60

    # Overwrite
    box.top_left = new_pt

    # Cache should be busted, returning new values
    assert box.top_left.x == 50
    assert box.top_left.y == 60


def test_union_assignment(assignment_ffi):
    """
    Tests assigning structs into a union field, verifying that overlapping
    memory behaves correctly.
    """
    ffi = assignment_ffi
    vu = ffi.new("union VectorUnion")

    # Write a 3D point
    p3 = ffi.new("struct Point3D")
    p3.x = 1
    p3.y = 2
    p3.z = 3
    vu.p3 = p3

    assert vu.p3.x == 1
    assert vu.p3.z == 3
    
    # Overwrite the overlapping 2D point portion
    p2 = ffi.new("struct Point2D")
    p2.x = 99
    p2.y = 88
    vu.p2 = p2

    # Verify the 2D fields updated, but the Z field from the 3D point remains intact
    assert vu.p2.x == 99
    assert vu.p3.x == 99
    assert vu.p3.y == 88
    assert vu.p3.z == 3  # Untouched by the p2 assignment


def test_struct_assignment_size_mismatch(assignment_ffi):
    """
    Ensures rigorous size checks prevent buffer overflows when attempting
    to assign structs of differing sizes.
    """
    ffi = assignment_ffi
    box = ffi.new("struct BoundingBox")
    p3 = ffi.new("struct Point3D")

    # Attempt to assign a 12-byte struct to an 8-byte field
    with pytest.raises(ValueError, match="Size mismatch: cannot assign struct of size 12.*size 8"):
        box.top_left = p3

    vu = ffi.new("union VectorUnion")
    p2 = ffi.new("struct Point2D")
    
    # Attempt to assign an 8-byte struct to a 12-byte union field
    with pytest.raises(ValueError, match="Size mismatch: cannot assign struct of size 8.*size 12"):
        vu.p3 = p2

def test_struct_assignment_offset_safety(assignment_ffi):
    """
    Ensures that assigning a struct to a field in the middle of another struct
    does not overwrite the fields before or after it.
    """
    ffi = assignment_ffi
    padded = ffi.new("struct PaddedStruct")
    
    # Using 0x7AAAAAAA instead of 0xAAAAAAAA to stay within 
    # positive bounds of a signed 32-bit C-integer.
    padded.before = 0x7AAAAAAA
    padded.after = 0x7BBBBBBB

    pt = ffi.new("struct Point2D")
    pt.x = 100
    pt.y = 200

    # Assign to the middle struct
    padded.pt = pt

    # Verify the struct fields were copied
    assert padded.pt.x == 100
    assert padded.pt.y == 200

    # Verify bounds weren't overwritten
    assert padded.before == 0x7AAAAAAA
    assert padded.after == 0x7BBBBBBB


def test_struct_assignment_raw_bytes(assignment_ffi):
    """
    Verifies that assigning a correctly sized byte-like object to a struct
    field successfully updates the underlying memory.
    """
    import struct
    ffi = assignment_ffi
    box = ffi.new("struct BoundingBox")
    
    # Point2D is two 32-bit ints (x, y). Let's pack x=42, y=84 in little-endian.
    raw_data = struct.pack("<ii", 42, 84)
    
    # Assign raw bytes directly
    box.top_left = raw_data
    
    # Verify it unpacked correctly through DFFI's struct interpretation
    assert box.top_left.x == 42
    assert box.top_left.y == 84


def test_struct_assignment_invalid_types(assignment_ffi):
    """
    Verifies that assignment strictly requires a BoundTypeInstance or byte-like
    object, and rejects primitive Python types or incorrectly sized bytes.
    """
    ffi = assignment_ffi
    box = ffi.new("struct BoundingBox")

    with pytest.raises(TypeError, match="Cannot assign int to struct/union field"):
        box.top_left = 123

    with pytest.raises(TypeError, match="Cannot assign str to struct/union field"):
        box.top_left = "invalid"

    # Bytes are allowed, but incorrectly sized bytes must still fail
    with pytest.raises(ValueError, match="Size mismatch: expected 8 bytes, got 4."):
        box.top_left = b"\x00" * 4