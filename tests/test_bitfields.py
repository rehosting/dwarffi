import pytest
from dwarffi.core import isf_from_dict

def test_bitfield_read_write(base_types_little_endian):
    # Simulate:
    # struct flags {
    #     int flag_a : 3;
    #     int flag_b : 5;
    # };
    isf = isf_from_dict({
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
    inst = isf.create_instance("bitfield_struct", buf)

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