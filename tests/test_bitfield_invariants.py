import struct

from dwarffi import DFFI


def test_bitfield_preserves_other_bits_across_patterns():
    base_types = {
        "u16": {"size": 2, "signed": False, "kind": "int", "endian": "little"},
        "void": {"size": 0, "signed": False, "kind": "void", "endian": "little"},
        "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"},
    }
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types,
        "user_types": {
            "bf": {
                "kind": "struct", "size": 2,
                "fields": {
                    # 5-bit field at bit 3
                    "f": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 5, "bit_position": 3,
                                               "type": {"kind": "base", "name": "u16"}}},
                },
            }
        },
        "enums": {}, "symbols": {},
    })

    mask = ((1 << 5) - 1) << 3  # 0b11111 << 3
    patterns = [0x0000, 0xFFFF, 0xA55A, 0x1234, 0x8001, 0x0F0F]

    for cur in patterns:
        buf = bytearray(2)
        struct.pack_into("<H", buf, 0, cur)
        inst = ffi.from_buffer("struct bf", buf)

        inst.f = 0  # clear field bits only
        out = struct.unpack_from("<H", buf, 0)[0]
        assert out == (cur & ~mask)

        inst.f = 0x1F  # set all field bits
        out2 = struct.unpack_from("<H", buf, 0)[0]
        assert out2 == ((cur & ~mask) | mask)
