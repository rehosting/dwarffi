import pytest

from dwarffi import DFFI


def test_struct_char_array_assignment():
    """
    Verifies direct assignment of string and byte values to character array fields.
    Tests zero-filling, truncation (leaving space for null), and various input types.
    """
    isf = {
        "metadata": {},
        "base_types": {
            "char": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
        },
        "user_types": {
            "test_struct": {
                "kind": "struct", "size": 20,
                "fields": {
                    "name": {
                        "offset": 0, 
                        "type": {"kind": "array", "count": 10, "subtype": {"kind": "base", "name": "char"}}
                    },
                    "data": {
                        "offset": 10, 
                        "type": {"kind": "array", "count": 5, "subtype": {"kind": "base", "name": "char"}}
                    },
                    "ints": {
                        "offset": 15, 
                        "type": {"kind": "array", "count": 1, "subtype": {"kind": "base", "name": "int"}}
                    }
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    ffi = DFFI(isf)
    inst = ffi.new("struct test_struct")
    
    # 1. Successful string assignment (fits in buffer)
    # Buffer is 10 bytes. "hello" fits, gets null terminated and zero-filled.
    inst.name = "hello"
    assert ffi.string(inst.name) == b"hello"
    assert bytes(inst)[:10] == b"hello\x00\x00\x00\x00\x00"

    # 2. String truncation (enforces null terminator)
    # Buffer is 5 bytes. Data "12345" should truncate to "1234" + "\0".
    inst.data = "12345"
    assert ffi.string(inst.data) == b"1234"
    assert bytes(inst)[10:15] == b"1234\x00"

    # 3. Bytes and Bytearray assignment
    inst.name = b"bytes"
    assert ffi.string(inst.name) == b"bytes"
    assert bytes(inst)[:10] == b"bytes\x00\x00\x00\x00\x00"
    
    inst.name = bytearray(b"array")
    assert ffi.string(inst.name) == b"array"
    assert bytes(inst)[:10] == b"array\x00\x00\x00\x00\x00"

    # 4. Error: Direct assignment to non-byte arrays (e.g., int[1])
    # The implementation raises NotImplementedError if elem_size != 1.
    with pytest.raises(NotImplementedError, match="Direct assignment to non-byte array field"):
        inst.ints = b"\x01\x00\x00\x00"

    # 5. Error: Invalid input type for array field
    # The implementation now raises a TypeError for invalid types like int
    with pytest.raises(TypeError, match="Cannot assign int to array field 'name'"):
        inst.name = 12345

def test_struct_zero_length_array_assignment():
    """Verifies that assignment to a zero-length array returns early without error."""
    isf = {
        "metadata": {},
        "base_types": {"char": {"kind": "int", "size": 1, "signed": False, "endian": "little"}},
        "user_types": {
            "empty_arr": {
                "kind": "struct", "size": 4,
                "fields": {
                    "pad": {"offset": 0, "type": {"kind": "base", "name": "char"}},
                    "flex": {"offset": 1, "type": {"kind": "array", "count": 0, "subtype": {"kind": "base", "name": "char"}}}
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    ffi = DFFI(isf)
    inst = ffi.new("struct empty_arr")
    
    # Should return early since count <= 0.
    inst.flex = "nothing"
    assert bytes(inst) == b"\x00\x00\x00\x00"

def test_struct_direct_assignment():
    """
    Verifies direct assignment of a BoundTypeInstance (struct) to a struct field.
    Tests successful memory copies, cache invalidation, size mismatch errors, and type errors.
    """
    isf = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
        },
        "user_types": {
            "Point": {
                "kind": "struct", "size": 8,
                "fields": {
                    "x": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "y": {"offset": 4, "type": {"kind": "base", "name": "int"}}
                }
            },
            "Line": {
                "kind": "struct", "size": 16,
                "fields": {
                    "start": {"offset": 0, "type": {"kind": "struct", "name": "Point"}},
                    "end":   {"offset": 8, "type": {"kind": "struct", "name": "Point"}}
                }
            },
            "Point3D": {
                "kind": "struct", "size": 12,
                "fields": {
                    "x": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "y": {"offset": 4, "type": {"kind": "base", "name": "int"}},
                    "z": {"offset": 8, "type": {"kind": "base", "name": "int"}}
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    
    ffi = DFFI(isf)
    
    # Setup instances
    p1 = ffi.new("struct Point")
    p1.x = 10
    p1.y = 20
    
    p2 = ffi.new("struct Point")
    p2.x = 100
    p2.y = 200
    
    line = ffi.new("struct Line")
    
    # 1. Successful struct-to-struct assignment
    line.start = p1
    line.end = p2
    
    assert line.start.x == 10
    assert line.start.y == 20
    assert line.end.x == 100
    assert line.end.y == 200
    
    # Ensure memory matches exactly
    assert bytes(line.start) == bytes(p1)
    assert bytes(line.end) == bytes(p2)
    
    # 2. Test cache invalidation (overwrite start with p2)
    line.start = p2
    assert line.start.x == 100
    assert line.start.y == 200
    assert bytes(line.start) == bytes(p2)
    
    # 3. Error: Size mismatch
    p3d = ffi.new("struct Point3D")
    with pytest.raises(ValueError, match="Size mismatch: cannot assign struct of size 12"):
        line.start = p3d
        
    # 4. Error: Invalid type assignment
    with pytest.raises(TypeError, match="Cannot assign int to struct/union field"):
        line.start = 1234
        
    with pytest.raises(TypeError, match="Cannot assign str to struct/union field"):
        line.start = "not a struct"