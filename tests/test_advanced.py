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
    base_addr = adv_ffi_env.offset(outer)

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

def test_array_out_of_bounds_and_negative_indices(adv_ffi_env):
    # Array of 5 integers
    arr = adv_ffi_env.new("int[5]", [1, 2, 3, 4, 5])

    # Standard bounds checks
    with pytest.raises(IndexError, match="out of bounds"):
        arr[5]
    with pytest.raises(IndexError, match="out of bounds"):
        arr[5] = 100

    # Negative indices (currently not explicitly supported as Python lists do, 
    # but they should raise IndexError rather than quietly corrupting memory)
    with pytest.raises(IndexError, match="out of bounds"):
        arr[-1]
    with pytest.raises(IndexError, match="out of bounds"):
        arr[-1] = 99
        
    # Valid assignments don't raise
    arr[4] = 99
    assert arr[4] == 99

def test_nested_array_of_structs(adv_ffi_env):
    # Create an array of 3 'outer_struct' instances
    # (Each outer_struct is 12 bytes: 4 for inner.val, 8 for arr[2])
    arr = adv_ffi_env.new("outer_struct[3]")
    
    # Test Deep Write via indexing
    arr[0].inner.val = 100
    arr[0].arr[1] = 999
    
    arr[2].inner.val = 300
    arr[2].arr[0] = 777
    
    # Test deep read
    assert arr[0].inner.val == 100
    assert arr[0].arr[1] == 999
    assert arr[2].inner.val == 300
    assert arr[2].arr[0] == 777
    
    # Verify total array size
    assert adv_ffi_env.sizeof(arr) == 36 # 3 elements * 12 bytes each
    
    # Test slice extraction of inner structs natively converts to list of BoundTypeInstances
    slice_out = arr[0:3]
    assert len(slice_out) == 3
    assert slice_out[0].inner.val == 100
    assert slice_out[2].arr[0] == 777

def test_nested_anonymous_complex(base_types_little_endian):
    """Tests highly nested anonymous unions/structs (common in SoC register maps)."""
    from dwarffi.dffi import DFFI
    
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types_little_endian,
        "user_types": {
            "ctrl_reg": {
                "kind": "struct", "size": 4,
                "fields": {
                    "raw": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "bits": {
                        "offset": 0, "anonymous": True,
                        "type": {"kind": "struct", "name": "bits_layout"}
                    }
                }
            },
            "bits_layout": {
                "kind": "struct", "size": 4,
                "fields": {
                    "enabled": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 1, "bit_position": 0, "type": {"kind": "base", "name": "int"}}},
                    "mode": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 3, "bit_position": 1, "type": {"kind": "base", "name": "int"}}}
                }
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {}
    })

    reg = ffi.from_buffer("struct ctrl_reg", bytearray(4))
    
    # Test flattened access through multiple layers of anonymity
    reg.mode = 7 # 0b111
    reg.enabled = 1
    
    # 0b111 (mode) << 1 | 1 (enabled) = 0b1111 = 0xF
    assert reg.raw == 0xF
    
    # Test writing to 'raw' updates flattened bits
    reg.raw = 0x0
    assert reg.mode == 0
    assert reg.enabled == 0

def test_union_in_union_overlap(adv_ffi_env):
    # struct { union { uint32_t a; union { uint16_t b; uint8_t c; } } }
    # All members (a, b, c) start at offset 0.
    union_isf = {
        "metadata": {},
        "base_types": {
            "u32": {"kind": "int", "size": 4, "signed": False, "endian": "little"},
            "u16": {"kind": "int", "size": 2, "signed": False, "endian": "little"},
            "u8": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
        },
        "user_types": {
            "nested_union": {
                "kind": "union", "size": 4,
                "fields": {
                    "a": {"offset": 0, "type": {"kind": "base", "name": "u32"}},
                    "inner": {"offset": 0, "type": {"kind": "union", "name": "inner_union"}}
                }
            },
            "inner_union": {
                "kind": "union", "size": 2,
                "fields": {
                    "b": {"offset": 0, "type": {"kind": "base", "name": "u16"}},
                    "c": {"offset": 0, "type": {"kind": "base", "name": "u8"}}
                }
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {}
    }
    adv_ffi_env.load_isf(union_isf)
    
    u = adv_ffi_env.new("struct nested_union")
    u.a = 0xAABBCCDD
    
    # Test overlap
    assert u.inner.b == 0xCCDD
    assert u.inner.c == 0xDD
    
    # Writing to inner affects outer
    u.inner.c = 0xFF
    assert u.a == 0xAABBCCFF
