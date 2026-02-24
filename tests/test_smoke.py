import struct

from dwarffi.dffi import DFFI


def test_load_isf_json_filelike_and_basic_read_write() -> None:
    isf_dict = {
        "metadata": {},
        "base_types": {
            "int": {"size": 4, "signed": True, "kind": "int", "endian": "little"},
            "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"},
            "void": {"size": 0, "signed": False, "kind": "void", "endian": "little"},
        },
        "user_types": {
            "my_struct": {
                "kind": "struct", "size": 16,
                "fields": {
                    "id": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "ptr": {
                        "offset": 8,
                        "type": {"kind": "pointer", "subtype": {"kind": "base", "name": "void"}},
                    },
                },
            }
        },
        "enums": {},
        "symbols": {},
    }

    ffi = DFFI(isf_dict)

    struct_size = ffi.sizeof("struct my_struct")
    assert struct_size == 16

    buf = bytearray(struct_size)
    struct.pack_into("<i", buf, 0, 123)
    struct.pack_into("<Q", buf, 8, 0x1122334455667788)

    inst = ffi.from_buffer("struct my_struct", buf)
    assert inst.id == 123
    assert inst.ptr.address == 0x1122334455667788

    inst.id = 456
    assert struct.unpack_from("<i", buf, 0)[0] == 456

    int_buf = bytearray(4)
    int_inst = ffi.from_buffer("int", int_buf)
    int_inst[0] = 7
    assert int_inst[0] == 7
