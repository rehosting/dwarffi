import pytest

from dwarffi import DFFI
from dwarffi.backend import BytesBackend


@pytest.fixture
def array_ffi():
    isf = {
        "metadata": {},
        "base_types": {
            "char": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
        },
        "user_types": {
            "nested_anon": {
                "kind": "struct", "size": 10,
                "fields": {
                    "un": {
                        "offset": 0, "anonymous": True, 
                        "type": {"kind": "union", "name": "inner_union"}
                    }
                }
            },
            "inner_union": {
                "kind": "union", "size": 10,
                "fields": {
                    "buf": {"offset": 0, "type": {"kind": "array", "count": 10, "subtype": {"kind": "base", "name": "char"}}}
                }
            },
            "multi_dim": {
                "kind": "struct", "size": 10,
                "fields": {
                    "grid": {"offset": 0, "type": {"kind": "array", "count": 2, "subtype": {"kind": "array", "count": 5, "subtype": {"kind": "base", "name": "char"}}}}
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    return DFFI(isf)

def test_assignment_to_nested_anonymous_array(array_ffi):
    """Verifies that direct assignment works through anonymous nested unions/structs."""
    inst = array_ffi.new("struct nested_anon")
    
    # Accessing 'buf' which is inside an anonymous union
    inst.buf = "nested"
    
    assert array_ffi.string(inst.buf) == b"nested"
    assert bytes(inst) == b"nested\x00\x00\x00\x00"

# tests/test_struct_array_assignment_advanced.py

def test_assignment_with_backend_proxy(array_ffi):
    """Verifies that assignment works correctly when backed by a MemoryBackend (LiveMemoryProxy)."""
    storage = bytearray(b"\xFF" * 20)
    array_ffi.backend = BytesBackend(storage)

    # Bind to address 0x5 in the backend
    inst = array_ffi.from_address("struct nested_anon", 0x5)

    inst.buf = "live"
    print("storage[0:20] =", bytes(storage[0:20]))
    print("storage[5:15] =", bytes(storage[5:15]))
    print("storage[0:10] =", bytes(storage[0:10]))


    # Verify the backend storage was updated correctly
    assert storage[5:15] == b"live\x00\x00\x00\x00\x00\x00"

def test_rejection_of_multidimensional_arrays(array_ffi):
    """
    Verifies that direct assignment is rejected for multidimensional arrays.
    The subtype of 'grid' (char[2][5]) is 'char[5]', which has a size of 5, not 1.
    """
    inst = array_ffi.new("struct multi_dim")
    
    # Should raise NotImplementedError because elem_size (5) != 1
    with pytest.raises(NotImplementedError, match="Direct assignment to non-byte array field"):
        inst.grid = "too_complex"

def test_assignment_input_types(array_ffi):
    """Tests assignment using memoryview and large input truncation."""
    inst = array_ffi.new("struct nested_anon")
    
    # 1. Test memoryview
    mv = memoryview(b"memview")
    inst.buf = mv
    assert array_ffi.string(inst.buf) == b"memview"
    
    # 2. Test large input (15 chars into 10-byte buffer)
    # Result should be 9 chars + 1 null terminator
    inst.buf = "0123456789ABCDE"
    assert array_ffi.string(inst.buf) == b"012345678"
    assert bytes(inst.buf) == b"012345678\x00"

def test_cache_consistency_after_assignment(array_ffi):
    """Ensures the BoundArrayView is properly invalidated and recreated after assignment."""
    inst = array_ffi.new("struct nested_anon")
    
    # 1. Initial access caches a BoundArrayView
    view1 = inst.buf
    assert bytes(view1) == b"\x00" * 10
    
    # 2. Assigning to the field triggers 'del self._instance_cache[name]'
    inst.buf = "rebuild"
    
    # 3. Next access should yield a new view with updated data
    view2 = inst.buf
    assert view2 is not view1
    assert array_ffi.string(view2) == b"rebuild"

def test_array_advanced_assignment():
    """
    Verifies that Python lists, tuples, and other BoundArrayViews can be 
    directly assigned to array fields natively in DFFI.
    """
    isf = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
        },
        "user_types": {
            "Point2D": {
                "kind": "struct", "size": 8,
                "fields": {
                    "x": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "y": {"offset": 4, "type": {"kind": "base", "name": "int"}}
                }
            },
            "ArrayContainer": {
                "kind": "struct", "size": 64,
                "fields": {
                    "numbers": {"offset": 0, "type": {"kind": "array", "count": 5, "subtype": {"kind": "base", "name": "int"}}},
                    "more_numbers": {"offset": 20, "type": {"kind": "array", "count": 5, "subtype": {"kind": "base", "name": "int"}}},
                    "points": {"offset": 40, "type": {"kind": "array", "count": 3, "subtype": {"kind": "struct", "name": "Point2D"}}}
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    ffi = DFFI(isf)
    inst = ffi.new("struct ArrayContainer")

    # 1. Assign List of Integers
    inst.numbers = [10, 20, 30]
    assert inst.numbers[0] == 10
    assert inst.numbers[1] == 20
    assert inst.numbers[2] == 30
    assert inst.numbers[3] == 0  # Should remain unmodified

    # 2. Assign Tuple of Integers (Partially overwrites)
    inst.numbers = (100, 200)
    assert inst.numbers[0] == 100
    assert inst.numbers[1] == 200
    assert inst.numbers[2] == 30 # Retained from list assignment
    
    # 3. Assign BoundArrayView (Fast copy between fields)
    inst.more_numbers = inst.numbers
    assert inst.more_numbers[0] == 100
    assert inst.more_numbers[1] == 200
    assert inst.more_numbers[2] == 30

    # 4. Assign List of Structs
    p1 = ffi.new("struct Point2D")
    p1.x, p1.y = 1, 1
    p2 = ffi.new("struct Point2D")
    p2.x, p2.y = 2, 2
    
    inst.points = [p1, p2]
    assert inst.points[0].x == 1
    assert inst.points[1].y == 2
    assert inst.points[2].x == 0 # Unmodified struct

    # 5. Error Cases
    with pytest.raises(ValueError, match="Cannot assign sequence of size 6"):
        inst.numbers = [1, 2, 3, 4, 5, 6]
        
    with pytest.raises(TypeError, match="Element size mismatch"):
        # Numbers uses element size 4, points uses element size 8
        inst.points = inst.numbers  

    with pytest.raises(TypeError, match="Cannot assign int to array field"):
        inst.numbers = 123