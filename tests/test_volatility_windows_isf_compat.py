import json
from pathlib import Path

from dwarffi import DFFI


FIXTURE = Path(__file__).parent / "fixtures" / "windows_min_ntkrnlmp.json"


def _ptr_expected(value: int, ptr_size: int) -> int:
    assert ptr_size in (4, 8)
    return value & (0xFFFFFFFF if ptr_size == 4 else 0xFFFFFFFFFFFFFFFF)


def test_can_load_volatility_style_windows_isf_and_resolve_structs():
    isf = json.loads(FIXTURE.read_text("utf-8"))
    ffi = DFFI(isf)

    # Schema shape sanity checks
    assert "base_types" in isf and "user_types" in isf and "symbols" in isf
    assert "pointer" in isf["base_types"], "Windows ISF fixture must include base_types['pointer']"

    le = ffi.typeof("struct _LIST_ENTRY")
    us = ffi.typeof("struct _UNICODE_STRING")

    assert ffi.sizeof(le) > 0
    assert ffi.sizeof(us) > 0


def test_instance_field_io_and_serialization_helpers():
    isf = json.loads(FIXTURE.read_text("utf-8"))
    ffi = DFFI(isf)

    ptr_size = isf["base_types"]["pointer"]["size"]

    # Use _UNICODE_STRING because it contains primitive fields + a pointer
    # Typical fields: Length, MaximumLength, Buffer (PWSTR)
    us = ffi.typeof("struct _UNICODE_STRING")
    buf = bytearray(ffi.sizeof(us))
    inst = ffi.from_buffer("struct _UNICODE_STRING", buf)

    # Field presence (names are extremely stable in Windows PDB-derived types)
    assert hasattr(inst, "Length")
    assert hasattr(inst, "MaximumLength")
    assert hasattr(inst, "Buffer")

    inst.Length = 8
    inst.MaximumLength = 16

    raw_ptr = 0x1122334455667788
    inst.Buffer = raw_ptr

    # Pointer readback should respect ptr width (32-bit truncation is expected)
    assert int(inst.Buffer) == _ptr_expected(raw_ptr, ptr_size)

    # ---- to_dict() demo: confirms recursive conversion + pointer->address
    d = ffi.to_dict(inst)
    assert d["Length"] == 8
    assert d["MaximumLength"] == 16
    assert d["Buffer"] == _ptr_expected(raw_ptr, ptr_size)

    # ---- pretty_print() demo: confirms recursive display and pointer repr is included
    s = ffi.pretty_print(inst)
    assert "_UNICODE_STRING" in s
    assert "Length" in s
    assert "MaximumLength" in s
    assert "Buffer" in s
    # We don’t assert the exact repr formatting, but we DO assert the address appears.
    assert hex(_ptr_expected(raw_ptr, ptr_size)).lower().replace("0x", "") in s.lower()

    # Also demonstrate LIST_ENTRY because it's canonical for Windows kernel ISFs
    le = ffi.typeof("struct _LIST_ENTRY")
    le_buf = bytearray(ffi.sizeof(le))
    le_inst = ffi.from_buffer("struct _LIST_ENTRY", le_buf)

    assert hasattr(le_inst, "Flink")
    assert hasattr(le_inst, "Blink")

    flink_raw = 0xAABBCCDDEEFF0011
    blink_raw = 0x1100FFEEDDCCBBAA
    le_inst.Flink = flink_raw
    le_inst.Blink = blink_raw

    assert int(le_inst.Flink) == _ptr_expected(flink_raw, ptr_size)
    assert int(le_inst.Blink) == _ptr_expected(blink_raw, ptr_size)

    le_dict = ffi.to_dict(le_inst)
    assert le_dict["Flink"] == _ptr_expected(flink_raw, ptr_size)
    assert le_dict["Blink"] == _ptr_expected(blink_raw, ptr_size)


def test_inspect_layout_prints_offsets_and_types(capsys):
    isf = json.loads(FIXTURE.read_text("utf-8"))
    ffi = DFFI(isf)

    ffi.inspect_layout("struct _UNICODE_STRING")
    out = capsys.readouterr().out

    # Layout header
    assert "Layout of struct _UNICODE_STRING" in out
    assert "Offset" in out and "Size" in out and "Field" in out and "Type" in out

    # Field names should appear
    assert "Length" in out
    assert "MaximumLength" in out
    assert "Buffer" in out

    # Pointer type name should appear as kind or name (depends on ISF / your formatter)
    # We accept either because inspect_layout uses kind/name fallback.
    assert ("pointer" in out) or ("Pointer" in out) or ("Ptr" in out)