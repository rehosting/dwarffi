import json
import pytest
from dwarffi.dffi import DFFI


@pytest.fixture
def adv_ffi_env(tmp_path):
    """
    Creates an ISF definition specifically designed to test advanced nested layouts,
    anonymous unions/structs, and C11-style memory maps.
    """
    isf_data = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "short": {"kind": "int", "size": 2, "signed": False, "endian": "little"},
            "char": {"kind": "char", "size": 1, "signed": False, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
        },
        "user_types": {
            "inner_struct": {
                "kind": "struct",
                "size": 4,
                "fields": {"val": {"offset": 0, "type": {"kind": "base", "name": "int"}}},
            },
            "outer_struct": {
                "kind": "struct",
                "size": 12,
                "fields": {
                    "inner": {"offset": 0, "type": {"kind": "struct", "name": "inner_struct"}},
                    "arr": {
                        "offset": 4,
                        "type": {
                            "kind": "array",
                            "count": 2,
                            "subtype": {"kind": "base", "name": "int"},
                        },
                    },
                },
            },
            # Anonymous Structs/Unions (Common in Hardware Registers)
            "anon_struct": {
                "kind": "struct",
                "size": 4,
                "fields": {
                    "LOW": {"offset": 0, "type": {"kind": "base", "name": "short"}},
                    "HIGH": {"offset": 2, "type": {"kind": "base", "name": "short"}},
                },
            },
            "anon_union": {
                "kind": "union",
                "size": 4,
                "fields": {
                    "ALL": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "__anon_1": {
                        "offset": 0,
                        "anonymous": True,
                        "type": {"kind": "struct", "name": "anon_struct"},
                    },
                },
            },
            "reg_map": {
                "kind": "struct",
                "size": 4,
                "fields": {
                    "__anon_2": {
                        "offset": 0,
                        "anonymous": True,
                        "type": {"kind": "union", "name": "anon_union"},
                    }
                },
            },
        },
        "enums": {},
        "symbols": {},
    }

    isf_file = tmp_path / "adv_test.isf.json"
    with open(isf_file, "w") as f:
        json.dump(isf_data, f)

    return DFFI(str(isf_file))


# ==============================================================================
# Tests
# ==============================================================================


def test_dynamic_arrays_and_slicing(adv_ffi_env: DFFI):
    # 1. Dynamically allocate an array not explicitly defined in the ISF
    arr = adv_ffi_env.new("int[5]", [10, 20, 30, 40, 50])

    # 2. Check length and access
    assert len(arr) == 5
    assert arr[2] == 30

    # 3. Test Slicing! This should return a list natively
    assert arr[1:4] == [20, 30, 40]

    # 4. Strings via char arrays
    s = adv_ffi_env.new("char[]", b"hello")
    assert len(s) == 6  # "hello" is 5 chars + 1 null terminator
    assert adv_ffi_env.string(s) == b"hello"


def test_deep_struct_initialization(adv_ffi_env: DFFI):
    # Pass a complex dictionary to initialize nested structures and arrays all at once
    outer = adv_ffi_env.new("outer_struct", {"inner": {"val": 42}, "arr": [100, 200]})

    # Verify the deep initialization cascaded into the bytearray correctly
    assert outer.inner.val == 42
    assert outer.arr[0] == 100
    assert outer.arr[1] == 200

    # Verify it can be read as a slice
    assert outer.arr[:] == [100, 200]


def test_anonymous_fields(adv_ffi_env: DFFI):
    # Initialize the register map
    reg = adv_ffi_env.new("struct reg_map", {"ALL": 0x12345678})

    # 1. Access the 'ALL' field (nested inside union __anon_2)
    assert reg.ALL == 0x12345678

    # 2. Access 'LOW' and 'HIGH' (nested inside struct __anon_1 inside union __anon_2)
    # Because this is Little Endian, the LOW short sits at byte 0, HIGH sits at byte 2
    assert reg.LOW == 0x5678
    assert reg.HIGH == 0x1234

    # 3. Write to an anonymous field and ensure it propagates up
    reg.LOW = 0xAAAA
    assert reg.ALL == 0x1234AAAA


def test_addressof_api_alignment(adv_ffi_env: DFFI):
    outer = adv_ffi_env.new("outer_struct", {"inner": {"val": 77}})

    # Base address of the instance (relative to its own bytearray is 0)
    base_addr = outer.offset

    # addressof() should now return a Ptr object, just like CFFI
    ptr_outer = adv_ffi_env.addressof(outer)
    assert ptr_outer.address == base_addr
    assert ptr_outer.points_to_type_name == "outer_struct"

    # Navigating into fields
    ptr_inner = adv_ffi_env.addressof(outer, "inner")
    assert ptr_inner.address == base_addr + 0
    assert ptr_inner.points_to_type_name == "inner_struct"

    ptr_val = adv_ffi_env.addressof(outer, "inner", "val")
    assert ptr_val.address == base_addr + 0
    assert ptr_val.points_to_type_name == "int"


def test_equality_operator(adv_ffi_env: DFFI):
    # Create two different instances with the exact same bytes
    inst1 = adv_ffi_env.new("inner_struct", {"val": 99})
    inst2 = adv_ffi_env.new("inner_struct", {"val": 99})

    # They should evaluate as equal natively
    assert inst1 == inst2

    # Modifying one should break equality
    inst2.val = 100
    assert inst1 != inst2

    # Different types with the same bytes are NOT equal
    int_inst = adv_ffi_env.new("int", 99)
    # Even though both 'int' and 'inner_struct' are 4 bytes with value 99
    assert inst1 != int_inst
