import io
import json
import struct

from dwarffi.core import VtypeJson, load_isf_json


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
                "kind": "struct",
                "size": 16,
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

    f = io.StringIO(json.dumps(isf_dict))
    isf = load_isf_json(f)
    assert isinstance(isf, VtypeJson)

    struct_def = isf.get_user_type("my_struct")
    assert struct_def is not None
    assert struct_def.size == 16

    buf = bytearray(struct_def.size)
    struct.pack_into("<i", buf, 0, 123)
    struct.pack_into("<Q", buf, 8, 0x1122334455667788)

    inst = isf.create_instance("my_struct", buf)
    assert inst.id == 123
    assert inst.ptr.address == 0x1122334455667788

    inst.id = 456
    assert struct.unpack_from("<i", buf, 0)[0] == 456

    int_buf = bytearray(4)
    int_inst = isf.create_instance("int", int_buf)
    int_inst._value = 7
    assert int_inst._value == 7
