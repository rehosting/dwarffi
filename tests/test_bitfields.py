from dwarffi import DFFI
import struct


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

def test_bitfield_crosses_byte_boundary_u16():
    # Storage unit is u16, and fields live across the byte boundary.
    # Layout (bit positions within the u16):
    #   low  : bits 0..3   (4 bits)
    #   mid  : bits 4..11  (8 bits)  <-- spans across the boundary
    #   high : bits 12..15 (4 bits)
    base_types = {
        "u16": {"size": 2, "signed": False, "kind": "int", "endian": "little"},
        "void": {"size": 0, "signed": False, "kind": "void", "endian": "little"},
        "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"},
    }
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types,
        "user_types": {
            "bf16": {
                "kind": "struct", "size": 2,
                "fields": {
                    "low":  {"offset": 0, "type": {"kind": "bitfield", "bit_length": 4, "bit_position": 0,
                                                  "type": {"kind": "base", "name": "u16"}}},
                    "mid":  {"offset": 0, "type": {"kind": "bitfield", "bit_length": 8, "bit_position": 4,
                                                  "type": {"kind": "base", "name": "u16"}}},
                    "high": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 4, "bit_position": 12,
                                                  "type": {"kind": "base", "name": "u16"}}},
                },
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {},
    })

    buf = bytearray(2)
    inst = ffi.from_buffer("struct bf16", buf)

    inst.low = 0xA     # 1010
    inst.mid = 0x5C    # 0101_1100 (spans bytes)
    inst.high = 0xB    # 1011

    assert inst.low == 0xA
    assert inst.mid == 0x5C
    assert inst.high == 0xB

    # Combined value: 0xB (high) << 12 | 0x5C (mid) << 4 | 0xA (low)
    expected = (0xB << 12) | (0x5C << 4) | 0xA
    assert struct.unpack_from("<H", buf, 0)[0] == expected


def test_bitfield_storage_unit_at_nonzero_struct_offset():
    # Ensure absolute_field_offset = instance_offset + field_offset is respected for bitfields.
    # struct {
    #   u8 pad0;       // offset 0
    #   u8 pad1;       // offset 1
    #   u16 storage;   // offset 2 (bitfields live here)
    # }
    base_types = {
        "u8": {"size": 1, "signed": False, "kind": "int", "endian": "little"},
        "u16": {"size": 2, "signed": False, "kind": "int", "endian": "little"},
        "void": {"size": 0, "signed": False, "kind": "void", "endian": "little"},
        "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"},
    }
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types,
        "user_types": {
            "bf_offset": {
                "kind": "struct", "size": 4,
                "fields": {
                    "pad0": {"offset": 0, "type": {"kind": "base", "name": "u8"}},
                    "pad1": {"offset": 1, "type": {"kind": "base", "name": "u8"}},
                    "a": {"offset": 2, "type": {"kind": "bitfield", "bit_length": 3, "bit_position": 0,
                                                "type": {"kind": "base", "name": "u16"}}},
                    "b": {"offset": 2, "type": {"kind": "bitfield", "bit_length": 5, "bit_position": 3,
                                                "type": {"kind": "base", "name": "u16"}}},
                },
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {},
    })

    buf = bytearray(4)
    buf[0] = 0xEE
    buf[1] = 0xFF

    inst = ffi.from_buffer("struct bf_offset", buf)
    inst.a = 5
    inst.b = 17

    assert inst.a == 5
    assert inst.b == 17
    assert buf[0] == 0xEE
    assert buf[1] == 0xFF

    expected_u16 = (17 << 3) | 5
    assert struct.unpack_from("<H", buf, 2)[0] == expected_u16


def test_bitfield_write_preserves_unrelated_bits_in_storage_unit():
    # The writer explicitly preserves other bits:
    # new = (current & ~mask) | ((val & mask) << pos) :contentReference[oaicite:3]{index=3}
    base_types = {
        "u16": {"size": 2, "signed": False, "kind": "int", "endian": "little"},
        "void": {"size": 0, "signed": False, "kind": "void", "endian": "little"},
        "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"},
    }
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types,
        "user_types": {
            "bf_preserve": {
                "kind": "struct", "size": 2,
                "fields": {
                    # Only touches bits 4..7
                    "nibble": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 4, "bit_position": 4,
                                                     "type": {"kind": "base", "name": "u16"}}},
                },
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {},
    })

    buf = bytearray(2)
    struct.pack_into("<H", buf, 0, 0xA55A)  # known pattern

    inst = ffi.from_buffer("struct bf_preserve", buf)
    inst.nibble = 0x0  # clear only bits 4..7

    out = struct.unpack_from("<H", buf, 0)[0]

    # Bits 4..7 should be cleared; everything else preserved.
    # Mask = 0xF << 4 = 0x00F0
    assert out == (0xA55A & ~0x00F0)


def test_bitfield_truncates_and_masks_negative_values():
    # Reads use: (storage >> pos) & mask :contentReference[oaicite:4]{index=4}
    # Writes use: (value & mask) << pos :contentReference[oaicite:5]{index=5}
    base_types = {
        "u8": {"size": 1, "signed": False, "kind": "int", "endian": "little"},
        "void": {"size": 0, "signed": False, "kind": "void", "endian": "little"},
        "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"},
    }
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types,
        "user_types": {
            "bf8": {
                "kind": "struct", "size": 1,
                "fields": {
                    "f": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 3, "bit_position": 0,
                                                "type": {"kind": "base", "name": "u8"}}},
                },
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {},
    })

    buf = bytearray(1)
    inst = ffi.from_buffer("struct bf8", buf)

    inst.f = 0xFF  # should truncate to 0b111 = 7
    assert inst.f == 7
    assert buf[0] & 0x07 == 0x07

    inst.f = -1  # -1 & 0b111 => 0b111
    assert inst.f == 7
    assert buf[0] & 0x07 == 0x07


def test_bitfield_big_endian_storage_unit_packing():
    # This documents current behavior: bit positions are interpreted on the integer value
    # obtained from the compiled base type, which is endian-aware. :contentReference[oaicite:6]{index=6}
    base_types = {
        "u16": {"size": 2, "signed": False, "kind": "int", "endian": "big"},
        "void": {"size": 0, "signed": False, "kind": "void", "endian": "big"},
        "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "big"},
    }
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types,
        "user_types": {
            "bf16_be": {
                "kind": "struct", "size": 2,
                "fields": {
                    "low":  {"offset": 0, "type": {"kind": "bitfield", "bit_length": 4, "bit_position": 0,
                                                  "type": {"kind": "base", "name": "u16"}}},
                    "high": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 4, "bit_position": 12,
                                                  "type": {"kind": "base", "name": "u16"}}},
                },
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {},
    })

    buf = bytearray(2)
    inst = ffi.from_buffer("struct bf16_be", buf)

    inst.low = 0xA
    inst.high = 0xB

    expected = (0xB << 12) | 0xA  # 0xB00A
    assert struct.unpack_from(">H", buf, 0)[0] == expected