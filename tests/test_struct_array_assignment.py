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
    with pytest.raises(NotImplementedError, match="Direct assignment to array field 'name' is not supported"):
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