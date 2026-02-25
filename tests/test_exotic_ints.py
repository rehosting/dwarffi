from dwarffi import DFFI


def test_exotic_integer_sizes():
    """
    Validates that DFFI smoothly handles sizes unsupported by Python's struct module
    (e.g., 16-byte integers and 3-byte integers) via the native integer fallback engine.
    """
    
    # We mock the base types manually so this test runs accurately on 
    # all host systems regardless of the local compiler's __int128 support.
    base_types = {
        "int128": {"size": 16, "signed": True, "kind": "int", "endian": "little"},
        "uint128": {"size": 16, "signed": False, "kind": "int", "endian": "little"},
        "int24": {"size": 3, "signed": True, "kind": "int", "endian": "little"},
    }
    
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types,
        "user_types": {
            "exotic_struct": {
                "kind": "struct", "size": 35,
                "fields": {
                    "huge_signed": {"offset": 0, "type": {"kind": "base", "name": "int128"}},
                    "huge_unsigned": {"offset": 16, "type": {"kind": "base", "name": "uint128"}},
                    "weird_size": {"offset": 32, "type": {"kind": "base", "name": "int24"}},
                }
            }
        },
        "enums": {}, "symbols": {}
    })
    
    buf = bytearray(35)
    inst = ffi.from_buffer("struct exotic_struct", buf)
    
    # -------------------------------
    # Test 128-bit signed integer
    # -------------------------------
    min_int128 = -(2**127)  # -170141183460469231731687303715884105728
    inst.huge_signed = min_int128
    assert inst.huge_signed == min_int128
    
    # Writing a positive that overflows should wrap natively
    inst.huge_signed = (2**127)  # Max + 1
    assert inst.huge_signed == min_int128
    
    # -------------------------------
    # Test 128-bit unsigned integer
    # -------------------------------
    max_uint128 = (2**128) - 1 # 340282366920938463463374607431768211455
    inst.huge_unsigned = max_uint128
    assert inst.huge_unsigned == max_uint128
    
    # Test truncation into unsigned layout
    inst.huge_unsigned = -1
    assert inst.huge_unsigned == max_uint128
    
    # -------------------------------
    # Test 24-bit signed integer
    # -------------------------------
    # A 3-byte signed int ranges from -8388608 to 8388607
    inst.weird_size = 8388607
    assert inst.weird_size == 8388607
    
    # Causing a 1-bit overflow should correctly sign-extend negative
    inst.weird_size = 8388608
    assert inst.weird_size == -8388608