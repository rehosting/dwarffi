import pytest

from dwarffi.dffi import DFFI


@pytest.fixture
def ffi_env(tmp_path):
    import json

    isf_data = {
        "metadata": {},
        "base_types": {
            "unsigned long": {"kind": "int", "size": 8, "signed": False, "endian": "little"},
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
        },
        "user_types": {
            "my_struct": {
                "kind": "struct",
                "size": 16,
                "fields": {
                    # Field using a primitive typedef
                    "count": {"offset": 0, "type": {"kind": "typedef", "name": "size_t"}},
                    # Field using a pointer typedef
                    "ptr": {"offset": 8, "type": {"kind": "typedef", "name": "int_ptr_t"}},
                },
            }
        },
        "typedefs": {
            "size_t": {"kind": "base", "name": "unsigned long"},
            "int_ptr_t": {"kind": "pointer", "subtype": {"kind": "base", "name": "int"}},
            "my_struct_t": {"kind": "struct", "name": "my_struct"},
            # Nested alias mapping to another typedef!
            "nested_td": {"kind": "typedef", "name": "size_t"},
        },
        "enums": {},
        "symbols": {},
    }
    isf_file = tmp_path / "test_typedef.json"
    with open(isf_file, "w") as f:
        json.dump(isf_data, f)
    return DFFI(str(isf_file))


def test_typedef_resolution_base(ffi_env: DFFI):
    # ffi.new("size_t") completely decays to an unsigned long
    val = ffi_env.new("size_t", 42)
    assert val[0] == 42
    assert ffi_env.sizeof("size_t") == 8

    # ffi.new("nested_td") decays through size_t down to unsigned long
    val2 = ffi_env.new("nested_td", 100)
    assert val2[0] == 100
    assert ffi_env.sizeof("nested_td") == 8


def test_typedef_resolution_pointer(ffi_env: DFFI):
    # Casting to int_ptr_t creates a Ptr object whose target type is just "int"
    ptr = ffi_env.cast("int_ptr_t", 0x1000)
    assert ptr.address == 0x1000
    assert ptr.points_to_type_name == "int"


def test_typedef_in_struct_fields(ffi_env: DFFI):
    # We can create a struct using its typedef alias
    s = ffi_env.new("my_struct_t")

    # And we can read/write fields that are masked by typedefs
    s.count = 999
    s.ptr = 0x2000

    assert s.count == 999
    assert s.ptr.address == 0x2000
    assert s.ptr.points_to_type_name == "int"
