from __future__ import annotations
import struct
from dwarffi.dffi import DFFI

def test_struct_field_write_and_to_bytes(base_types_little_endian) -> None:
    ffi = DFFI({
        "metadata": {},
        "base_types": base_types_little_endian,
        "user_types": {
            "my_struct": {
                "kind": "struct", "size": 8,
                "fields": {
                    "id": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "flags": {"offset": 4, "type": {"kind": "base", "name": "u8"}},
                },
            }
        },
        "enums": {}, "symbols": {},
    })

    buf = bytearray(8)
    struct.pack_into("<i", buf, 0, 100)
    struct.pack_into("<B", buf, 4, 1)

    inst = ffi.from_buffer("struct my_struct", buf)
    assert inst.id == 100
    assert inst.flags == 1

    inst.id = 999
    assert struct.unpack_from("<i", buf, 0)[0] == 999

    b = bytes(inst)
    assert isinstance(b, bytes)
    assert len(b) == 8

def test_base_type_instance_value_roundtrip(base_types_little_endian) -> None:
    ffi = DFFI({
        "metadata": {}, "base_types": base_types_little_endian,
        "user_types": {}, "enums": {}, "symbols": {},
    })

    int_buf = bytearray(4)
    struct.pack_into("<i", int_buf, 0, 12345)
    int_inst = ffi.from_buffer("int", int_buf)

    assert int_inst[0] == 12345
    assert int(int_inst) == 12345

def test_array_write_via_boundarrayview_and_cache_invalidation(base_types_little_endian) -> None:
    ffi = DFFI({
        "metadata": {}, "base_types": base_types_little_endian,
        "user_types": {
            "portal_ffi_call": {
                "kind": "struct", "size": 16,
                "fields": {
                    "args": {
                        "offset": 0,
                        "type": {"kind": "array", "count": 2, "subtype": {"kind": "base", "name": "u64"}},
                    }
                },
            }
        },
        "enums": {}, "symbols": {},
    })

    buf = bytearray(16)
    inst = ffi.from_buffer("struct portal_ffi_call", buf)

    view1 = inst.args
    view1[0] = 0xAAAAAAAAAAAAAAAA
    view1[1] = 0xBBBBBBBBBBBBBBBB

    assert struct.unpack_from("<Q", buf, 0)[0] == 0xAAAAAAAAAAAAAAAA
    assert struct.unpack_from("<Q", buf, 8)[0] == 0xBBBBBBBBBBBBBBBB

    # Because BoundArrayView.__setitem__ invalidates the parent's cache, a fresh access should rebuild.
    view2 = inst.args
    assert view2 is not view1
    assert view2[0] == 0xAAAAAAAAAAAAAAAA

def test_enum_field_read_write_and_base_enum_instance(base_types_little_endian) -> None:
    ffi = DFFI({
        "metadata": {}, "base_types": base_types_little_endian,
        "user_types": {
            "has_enum": {
                "kind": "struct", "size": 4,
                "fields": {"e": {"offset": 0, "type": {"kind": "enum", "name": "my_enum"}}},
            }
        },
        "enums": {"my_enum": {"size": 4, "base": "int", "constants": {"FOO": 1, "BAR": 2}}},
        "symbols": {},
    })

    buf = bytearray(4)
    struct.pack_into("<i", buf, 0, 1)
    inst = ffi.from_buffer("struct has_enum", buf)

    assert int(inst.e) == 1
    assert inst.e.name == "FOO"

    inst.e = "BAR"
    assert struct.unpack_from("<i", buf, 0)[0] == 2
    assert inst.e.name == "BAR"

    enum_buf = bytearray(4)
    struct.pack_into("<i", enum_buf, 0, 2)
    enum_inst = ffi.from_buffer("enum my_enum", enum_buf)
    assert enum_inst[0].name == "BAR"

    enum_inst[0] = "FOO"
    assert struct.unpack_from("<i", enum_buf, 0)[0] == 1

def test_direct_array_field_assignment_raises(base_types_little_endian) -> None:
    ffi = DFFI({
        "metadata": {}, "base_types": base_types_little_endian,
        "user_types": {
            "t": {
                "kind": "struct", "size": 16,
                "fields": {
                    "args": {
                        "offset": 0,
                        "type": {"kind": "array", "count": 2, "subtype": {"kind": "base", "name": "u64"}},
                    }
                },
            }
        },
        "enums": {}, "symbols": {},
    })

    buf = bytearray(16)
    inst = ffi.from_buffer("struct t", buf)
    inst.args[0] = 1
    inst.args[1] = 2
