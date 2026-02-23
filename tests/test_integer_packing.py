import pytest

from dwarffi.parser import isf_from_dict


@pytest.fixture
def ffi_env():
    """Generates an ISF containing all major integer sizes and signednesses."""
    return isf_from_dict(
        {
            "metadata": {},
            "base_types": {
                "u8": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
                "i8": {"kind": "int", "size": 1, "signed": True, "endian": "little"},
                "u16": {"kind": "int", "size": 2, "signed": False, "endian": "little"},
                "i16": {"kind": "int", "size": 2, "signed": True, "endian": "little"},
                "u32": {"kind": "int", "size": 4, "signed": False, "endian": "little"},
                "i32": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
                "u64": {"kind": "int", "size": 8, "signed": False, "endian": "little"},
                "i64": {"kind": "int", "size": 8, "signed": True, "endian": "little"},
            },
            "user_types": {},
            "enums": {},
            "symbols": {},
        }
    )


@pytest.mark.parametrize(
    "ctype, input_val, expected_val",
    [
        # 8-bit Unsigned tests
        ("u8", 0, 0),
        ("u8", 255, 255),
        ("u8", 256, 0),  # Overflow by 1
        ("u8", -1, 255),  # Underflow by 1
        ("u8", -5, 251),  # Negative
        ("u8", 0xFFFF, 255),  # Massive overflow
        # 8-bit Signed tests
        ("i8", 0, 0),
        ("i8", 127, 127),
        ("i8", 128, -128),  # Sign bit crossed
        ("i8", 255, -1),  # Max 8-bit unsigned is -1 signed
        ("i8", -128, -128),  # Min
        ("i8", -129, 127),  # Underflow
        ("i8", 0xFFFF, -1),  # Massive overflow resolving to -1
        # 16-bit Unsigned/Signed
        ("u16", -1, 65535),
        ("i16", 65535, -1),
        ("i16", 0xFFFFFFFF, -1),  # Overflows 32-bit bounds into a 16-bit
        # 32-bit limits (The specific issue previously encountered)
        ("u32", -1, 4294967295),
        ("u32", 0xFFFFFFFF, 4294967295),
        ("i32", 0xFFFFFFFF, -1),
        ("i32", 4294967296, 0),  # Exact overflow
        ("i32", -4294967297, -1),  # Massive negative underflow
        # 64-bit limits
        ("u64", -1, 0xFFFFFFFFFFFFFFFF),
        ("i64", 0xFFFFFFFFFFFFFFFF, -1),
        ("i64", 0x8000000000000000, -9223372036854775808),  # Min 64-bit signed
    ],
)
def test_integer_packing_boundaries(ffi_env, ctype, input_val, expected_val):
    # 1. Test Base Type direct assignment ([0])
    buf = bytearray(8)
    inst = ffi_env.create_instance(ctype, buf)

    inst[0] = input_val
    assert int(inst) == expected_val
    assert inst[0] == expected_val


def test_integer_packing_in_struct(ffi_env):
    """Ensure struct fields also properly wrap using the same logic."""
    # We dynamically add a struct to the ffi_env containing the types
    ffi_env._raw_user_types["test_struct"] = {
        "kind": "struct",
        "size": 5,
        "fields": {
            "a": {"offset": 0, "type": {"kind": "base", "name": "i32"}},
            "b": {"offset": 4, "type": {"kind": "base", "name": "u8"}},
        },
    }
    # Clear cache to recognize the new struct
    ffi_env._parsed_user_types_cache.clear()

    buf = bytearray(5)
    inst = ffi_env.create_instance("test_struct", buf)

    # Set massively out of bounds values
    inst.a = 0xFFFFFFFF  # Should wrap to -1 for i32
    inst.b = -5  # Should wrap to 251 for u8

    assert inst.a == -1
    assert inst.b == 251

    # Verify the bytearray reflects Little Endian packing exactly
    assert buf == b"\xff\xff\xff\xff\xfb"
