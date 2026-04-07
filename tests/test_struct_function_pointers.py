import pytest

from dwarffi import DFFI


@pytest.fixture
def mock_isf_with_function_pointer():
    """A mock ISF dictionary containing a struct with an anonymous function pointer."""
    return {
        "metadata": {"format": "6.2.0", "producer": {"name": "test", "version": "1.0"}},
        "base_types": {
            "int": {"size": 4, "signed": True, "kind": "int", "endian": "little"},
            "loff_t": {"size": 8, "signed": True, "kind": "int", "endian": "little"},
            "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"}
        },
        "user_types": {
            "file": {
                "kind": "struct",
                "size": 16,
                "fields": {
                    "f_pos": {"offset": 0, "type": {"kind": "base", "name": "loff_t"}}
                }
            },
            "file_operations": {
                "kind": "struct",
                "size": 8,
                "fields": {
                    "proc_lseek": {
                        "offset": 0,
                        "type": {
                            "kind": "pointer",
                            "subtype": {
                                "kind": "function",
                                "return_type": {"kind": "base", "name": "loff_t"},
                                "parameters": [
                                    {"kind": "pointer", "subtype": {"kind": "struct", "name": "file"}},
                                    {"kind": "base", "name": "loff_t"},
                                    {"name": "whence", "type": {"kind": "base", "name": "int"}}
                                ]
                            }
                        }
                    }
                }
            }
        },
        "enums": {},
        "symbols": {},
        "functions": {}
    }

def test_struct_function_pointer_introspection(mock_isf_with_function_pointer):
    """Verify that function pointers inside structs can be correctly introspected."""
    ffi = DFFI(mock_isf_with_function_pointer)

    # Create an instance
    fops = ffi.new("struct file_operations")
    ptr = fops.proc_lseek

    # Verify points_to_type_name
    assert ptr.points_to_type_name == "function"
    assert repr(ptr) == "<Ptr to function at 0x0>"

    # Verify signature extraction
    sig = ptr.signature
    assert sig is not None
    assert sig.return_type_info["name"] == "loff_t"

    # Verify parameters
    assert len(sig.args) == 3

    # First argument has no name, but valid type
    assert sig.args[0].name == ""
    assert sig.args[0].type_info["kind"] == "pointer"

    # Second argument has no name, but valid type
    assert sig.args[1].name == ""
    assert sig.args[1].type_info["name"] == "loff_t"

    # Third argument has both name and type
    assert sig.args[2].name == "whence"
    assert sig.args[2].type_info["name"] == "int"

def test_struct_function_pointer_legacy_compat():
    """Verify that an older ISF without return_type or parameters doesn't break."""
    isf = {
        "metadata": {"format": "6.2.0"},
        "base_types": {
            "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"}
        },
        "user_types": {
            "legacy_struct": {
                "kind": "struct",
                "size": 8,
                "fields": {
                    "old_func": {
                        "offset": 0,
                        "type": {
                            "kind": "pointer",
                            "subtype": {
                                "kind": "function"
                            }
                        }
                    }
                }
            }
        },
        "enums": {},
        "symbols": {},
    }
    ffi = DFFI(isf)
    inst = ffi.new("struct legacy_struct")

    ptr = inst.old_func
    assert ptr.points_to_type_name == "function"

    sig = ptr.signature
    assert sig is not None
    assert sig.return_type_info == {"kind": "void"}
    assert len(sig.args) == 0

def test_non_function_pointer_signature():
    """Verify that a non-function pointer returns None for signature."""
    isf = {
        "metadata": {"format": "6.2.0"},
        "base_types": {
            "int": {"size": 4, "signed": True, "kind": "int", "endian": "little"},
            "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"}
        },
        "user_types": {
            "my_struct": {
                "kind": "struct",
                "size": 8,
                "fields": {
                    "ptr": {
                        "offset": 0,
                        "type": {
                            "kind": "pointer",
                            "subtype": {
                                "kind": "base",
                                "name": "int"
                            }
                        }
                    }
                }
            }
        },
        "enums": {},
        "symbols": {},
    }
    ffi = DFFI(isf)
    inst = ffi.new("struct my_struct")

    ptr = inst.ptr
    assert ptr.points_to_type_name == "int"
    assert ptr.signature is None
