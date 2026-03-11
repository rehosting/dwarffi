import pytest
from dwarffi import DFFI

def _isf_base_le():
    return {
        "metadata": {},
        "base_types": {
            "u8": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
            "u32": {"kind": "int", "size": 4, "signed": False, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
            "void": {"kind": "void", "size": 0, "signed": False, "endian": "little"},
        },
        "user_types": {
            "blob": {
                "kind": "struct",
                "size": 8,
                "fields": {
                    "a": {"offset": 0, "type": {"kind": "base", "name": "u32"}},
                    "b": {"offset": 4, "type": {"kind": "base", "name": "u32"}},
                },
            }
        },
        "enums": {},
        "symbols": {},
    }

def test_zero_copy_views_share_bytes():
    ffi = DFFI(_isf_base_le())
    buf = bytearray(8)

    x = ffi.from_buffer("struct blob", buf, offset=0)
    y = ffi.from_buffer("u32[2]", buf, offset=0)

    x.a = 0x11223344
    assert y[0] == 0x11223344

    y[1] = 0xAABBCCDD
    assert x.b == 0xAABBCCDD

def test_require_writable_rejects_every_type():
    ffi = DFFI(_isf_base_le())
    ro = bytes(8)

    with pytest.raises(TypeError, match="Buffer is read-only"):
        ffi.from_buffer("struct blob", ro, require_writable=True)

    with pytest.raises(TypeError, match="Buffer is read-only"):
        ffi.from_buffer("u32[2]", ro, require_writable=True)

    with pytest.raises(TypeError, match="Buffer is read-only"):
        ffi.from_buffer("u32", ro[:4], require_writable=True)
