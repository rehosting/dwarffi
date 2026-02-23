import struct
import pytest
from dwarffi.parser import isf_from_dict

@pytest.fixture
def base_types_isf():
    return isf_from_dict({
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "float": {"kind": "float", "size": 4, "signed": True, "endian": "little"}
        },
        "user_types": {
            "dummy_struct": {"kind": "struct", "size": 4, "fields": {}}
        },
        "enums": {
            "status": {"size": 4, "base": "int", "constants": {"OK": 0, "ERROR": 1}}
        },
        "symbols": {}
    })

def test_pointer_dereference_syntax(base_types_isf):
    # Test [0] getter and setter on base type
    int_buf = bytearray(4)
    int_inst = base_types_isf.create_instance("int", int_buf)
    
    int_inst[0] = 500
    assert int_inst[0] == 500
    assert struct.unpack("<i", int_buf)[0] == 500
    
    # Test [0] getter/setter on Enum
    enum_buf = bytearray(4)
    enum_inst = base_types_isf.create_instance("status", enum_buf)
    
    enum_inst[0] = "ERROR"
    assert enum_inst[0].name == "ERROR"
    assert int(enum_inst[0]) == 1

def test_struct_dereference_error(base_types_isf):
    struct_buf = bytearray(4)
    struct_inst = base_types_isf.create_instance("dummy_struct", struct_buf)
    
    # In CFFI, struct[0] returns a reference to the struct itself 
    # to simulate deep copying/dereferencing a struct pointer.
    assert struct_inst[0] is struct_inst
    
    # Setting an entire struct via [0] is prevented by our API currently
    with pytest.raises(TypeError, match="Cannot overwrite entire struct"):
        struct_inst[0] = 123

def test_magic_type_conversions(base_types_isf):
    int_buf = bytearray(4)
    struct.pack_into("<i", int_buf, 0, 1024)
    int_inst = base_types_isf.create_instance("int", int_buf)
    
    # __int__
    assert int(int_inst) == 1024
    
    # __index__ (used for hex, oct, bin, and slicing)
    assert hex(int_inst) == "0x400"
    
    # __bool__
    assert bool(int_inst) is True
    int_inst[0] = 0
    assert bool(int_inst) is False

def test_float_conversion(base_types_isf):
    float_buf = bytearray(4)
    struct.pack_into("<f", float_buf, 0, 3.14159)
    float_inst = base_types_isf.create_instance("float", float_buf)
    
    # __float__
    assert pytest.approx(float(float_inst), 0.0001) == 3.14159

def test_invalid_conversions(base_types_isf):
    struct_buf = bytearray(4)
    struct_inst = base_types_isf.create_instance("dummy_struct", struct_buf)
    
    # You shouldn't be able to cast a raw struct to an int/float
    with pytest.raises(TypeError, match="Cannot convert struct/union"):
        int(struct_inst)
        
    with pytest.raises(TypeError, match="Cannot convert type"):
        float(struct_inst)