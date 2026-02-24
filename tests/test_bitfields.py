from dwarffi import DFFI


def test_bitfield_read_write(base_types_little_endian):
    # Simulate:
    # struct flags {
    #     int flag_a : 3;
    #     int flag_b : 5;
    # };
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types_little_endian,
        "user_types": {
            "bitfield_struct": {
                "kind": "struct",
                "size": 4,
                "fields": {
                    "flag_a": {
                        "offset": 0,
                        "type": {
                            "kind": "bitfield",
                            "bit_length": 3,
                            "bit_position": 0,
                            "type": {"kind": "base", "name": "int"}
                        }
                    },
                    "flag_b": {
                        "offset": 0,
                        "type": {
                            "kind": "bitfield",
                            "bit_length": 5,
                            "bit_position": 3,
                            "type": {"kind": "base", "name": "int"}
                        }
                    }
                }
            }
        },
        "enums": {},
        "symbols": {},
    })

    buf = bytearray(4)
    inst = ffi.from_buffer("struct bitfield_struct", buf)

    # Write to bitfields
    inst.flag_a = 5  # 101 in binary
    inst.flag_b = 10 # 01010 in binary

    # Verify they read back correctly
    assert inst.flag_a == 5
    assert inst.flag_b == 10

    # Verify the underlying byte packing:
    # flag_a takes bits 0-2 (101)
    # flag_b takes bits 3-7 (01010)
    # Combined: 01010101 binary = 0x55
    assert buf[0] == 0x55

def test_bitfield_truncation_safety(base_types_little_endian):
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types_little_endian,
        "user_types": {
            "bitfield_struct": {
                "kind": "struct", "size": 4,
                "fields": {
                    "small_flag": { 
                        "offset": 0,
                        "type": {"kind": "bitfield", "bit_length": 2, "bit_position": 0, "type": {"kind": "base", "name": "int"}}
                    },
                    "safe_flag": { 
                        "offset": 0,
                        "type": {"kind": "bitfield", "bit_length": 6, "bit_position": 2, "type": {"kind": "base", "name": "int"}}
                    }
                }
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {}
    })

    inst = ffi.from_buffer("struct bitfield_struct", bytearray(4))
    
    # Initialize safe_flag
    inst.safe_flag = 0b111111 # 63
    
    # Assign a value way too large for small_flag (e.g., 255 / 0xFF)
    # It should truncate to the bottom 2 bits (0b11 = 3)
    inst.small_flag = 255
    
    assert inst.small_flag == 3
    
    # Crucially, the adjacent bitfield should be entirely unaffected
    assert inst.safe_flag == 63