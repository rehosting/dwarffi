import pytest
import struct
from dwarffi import DFFI

def test_error_missing_pointer_definition():
    """Ensures a clear error when 'pointer' base type is missing but needed."""
    # ISF with an int but NO pointer definition
    isf_no_ptr = {
        "metadata": {},
        "base_types": {"int": {"kind": "int", "size": 4, "signed": True, "endian": "little"}},
        "user_types": {
            "has_ptr": {
                "kind": "struct", "size": 8,
                "fields": {
                    "p": {"offset": 0, "type": {"kind": "pointer", "subtype": {"kind": "base", "name": "int"}}}
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    
    ffi = DFFI(isf_no_ptr)
    
    # 1. sizeof() should raise KeyError for a pointer type
    with pytest.raises(KeyError, match="base type 'pointer' not found"):
        ffi.sizeof("int *")
        
    # 2. Bound access should raise KeyError when trying to read the pointer field
    inst = ffi.from_buffer("struct has_ptr", bytearray(8))
    with pytest.raises(KeyError, match="Base type 'pointer' not defined"):
        _ = inst.p

def test_error_missing_base_type_for_field():
    """Ensures a clear error when a field's underlying base type is missing."""
    isf_missing_base = {
        "metadata": {},
        "base_types": {}, # Empty base types
        "user_types": {
            "my_struct": {
                "kind": "struct", "size": 4,
                "fields": {"val": {"offset": 0, "type": {"kind": "base", "name": "u32"}}}
            }
        },
        "enums": {}, "symbols": {}
    }
    
    ffi = DFFI(isf_missing_base)
    inst = ffi.from_buffer("struct my_struct", bytearray(4))
    
    with pytest.raises(KeyError, match="Required base type 'u32' for field 'val' not found"):
        _ = inst.val

def test_error_missing_enum_base_type():
    """Ensures a clear error when an enum refers to a non-existent base type."""
    isf_missing_enum_base = {
        "metadata": {},
        "base_types": {}, # Missing 'int'
        "user_types": {},
        "enums": {
            "my_enum": {"size": 4, "base": "int", "constants": {"A": 1}}
        },
        "symbols": {}
    }
    
    ffi = DFFI(isf_missing_enum_base)
    inst = ffi.from_buffer("enum my_enum", bytearray(4))
    
    # Reading the enum requires the underlying 'int' definition to unpack bytes
    with pytest.raises(KeyError, match="Underlying base type 'int' for enum 'my_enum' not found"):
        _ = inst[0]

def test_error_missing_user_type_resolution():
    """Ensures a clear error when a struct field points to a missing user type."""
    isf_missing_struct = {
        "metadata": {},
        "base_types": {"int": {"kind": "int", "size": 4, "signed": True, "endian": "little"}},
        "user_types": {
            "outer": {
                "kind": "struct", "size": 4,
                "fields": {
                    "inner": {"offset": 0, "type": {"kind": "struct", "name": "missing_inner"}}
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    
    ffi = DFFI(isf_missing_struct)
    inst = ffi.from_buffer("struct outer", bytearray(4))
    
    with pytest.raises(KeyError, match="Struct/Union definition 'missing_inner' for field 'inner' not found"):
        _ = inst.inner