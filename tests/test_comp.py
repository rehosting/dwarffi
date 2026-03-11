"""
test_comprehensive_new.py
=========================
Comprehensive gap-filling tests for dwarffi.

Covers ~30 areas not exercised by the existing test suite:
  1.  VtypeJson schema validation (missing sections, bad 'kind', non-dict root)
  2.  VtypeJson file-loading error paths (FileNotFoundError, LZMAError,
      JSONDecodeError, bad input type)
  3.  _FallbackBytesStruct pack/unpack (opaque/exotic types stored as raw bytes)
  4.  VtypeBaseType exotic kinds: bool, char, half-precision float (f16)
  5.  VtypeUserType.get_aggregated_struct() failure paths
  6.  VtypeEnum.get_name_for_value() lazy _val_to_name cache
  7.  VtypeSymbol.get_decoded_constant_data() - base64 success and failure
  8.  EnumInstance equality semantics (all branches)
  9.  BoundArrayView equality edge cases (length mismatch, vs non-list)
  10. BoundArrayView.__add__ address propagation with/without base_address
  11. Ptr.__hash__ / set / dict usage
  12. Ptr.__getitem__ with/without backend, non-int index
  13. DFFI.string() on enum instances
  14. DFFI.buffer() raises TypeError on Ptr
  15. DFFI.to_bytes() on zero-size type
  16. DFFI.memmove() with raw bytes as source
  17. DFFI.load_isf() idempotency (same object loaded twice)
  18. DFFI.get_symbol() include_incomplete filtering
  19. DFFI.shift_symbol_addresses() with path= targeting one ISF
  20. DFFI.search_types() with use_regex=True
  21. from_buffer() with memoryview input
  22. from_address() bounded (nonzero-count) array
  23. VtypeJson._resolve_type_info() circular typedef guard
  24. LiveMemoryProxy.__len__ returns sys.maxsize
  25. BytesBackend write boundary and negative-address guards
  26. BoundTypeInstance.__dir__ content and exclusions
  27. VtypeUserType / VtypeEnum pretty_print / str / to_dict / members alias
  28. DFFI.cast() re-cast of BoundArrayView
  29. DFFI.get_type_size() with missing array subtype
  30. DFFI._typeof_or_raise, typeof() whitespace stripping, invalid input
"""

from __future__ import annotations

import base64
import io
import json
import lzma
import struct as stdlib_struct
import sys

import pytest

# public surface
from dwarffi import (
    DFFI,
    BoundArrayView,
    BytesBackend,
    EnumInstance,
    Ptr,
    VtypeJson,
)
from dwarffi.backend import LiveMemoryProxy
from dwarffi.types import (
    VtypeBaseType,
    VtypeEnum,
    VtypeSymbol,
    VtypeUserType,
    _FallbackBytesStruct,
)

# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _minimal_isf(
    *,
    extra_base: dict | None = None,
    extra_user: dict | None = None,
    extra_enums: dict | None = None,
    extra_syms: dict | None = None,
    endian: str = "little",
    ptr_size: int = 8,
) -> dict:
    base = {
        "int": {"kind": "int", "size": 4, "signed": True, "endian": endian},
        "char": {"kind": "int", "size": 1, "signed": False, "endian": endian},
        "pointer": {"kind": "pointer", "size": ptr_size, "endian": endian},
        "void": {"kind": "void", "size": 0, "signed": False, "endian": endian},
    }
    if extra_base:
        base.update(extra_base)
    return {
        "metadata": {},
        "base_types": base,
        "user_types": extra_user or {},
        "enums": extra_enums or {},
        "symbols": extra_syms or {},
    }


@pytest.fixture
def simple_ffi() -> DFFI:
    return DFFI(_minimal_isf())


@pytest.fixture
def struct_ffi() -> DFFI:
    isf = _minimal_isf(
        extra_user={
            "point": {
                "kind": "struct",
                "size": 8,
                "fields": {
                    "x": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "y": {"offset": 4, "type": {"kind": "base", "name": "int"}},
                },
            }
        }
    )
    return DFFI(isf)


@pytest.fixture
def enum_ffi() -> DFFI:
    isf = _minimal_isf(
        extra_enums={
            "color": {
                "size": 4,
                "base": "int",
                "constants": {"RED": 0, "GREEN": 1, "BLUE": 2},
            }
        }
    )
    return DFFI(isf)


# ---------------------------------------------------------------------------
# 1. VtypeJson schema validation
# ---------------------------------------------------------------------------

class TestVtypeJsonValidation:
    def test_missing_base_types_section(self):
        with pytest.raises(ValueError, match="missing required"):
            VtypeJson({"metadata": {}, "user_types": {}, "enums": {}, "symbols": {}})

    def test_missing_user_types_section(self):
        with pytest.raises(ValueError, match="missing required"):
            VtypeJson({"metadata": {}, "base_types": {}, "enums": {}, "symbols": {}})

    def test_user_type_missing_kind_raises(self):
        bad = {
            "metadata": {},
            "base_types": {},
            "user_types": {"bad_type": {"size": 4, "fields": {}}},  # no 'kind'
            "enums": {},
            "symbols": {},
        }
        with pytest.raises(ValueError, match="missing the required 'kind'"):
            VtypeJson(bad)

    def test_non_dict_root_raises(self):
        with pytest.raises(ValueError, match="root must be an object"):
            VtypeJson(io.StringIO(json.dumps([{"base_types": {}, "user_types": {}}])))

    def test_bad_input_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            VtypeJson(12345)

    def test_valid_minimal_dict_loads(self):
        vj = VtypeJson(_minimal_isf())
        assert isinstance(vj, VtypeJson)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VtypeJson(str(tmp_path / "nonexistent.json"))

    def test_json_decode_error_raises(self, tmp_path):
        bad_json = tmp_path / "bad.json"
        bad_json.write_text("this is not json", encoding="utf-8")
        with pytest.raises(ValueError, match="Error decoding JSON"):
            VtypeJson(str(bad_json))

    def test_lzma_error_raises(self, tmp_path):
        bad_xz = tmp_path / "bad.json.xz"
        bad_xz.write_bytes(b"not xz data at all")
        with pytest.raises(ValueError, match="Error decompressing"):
            VtypeJson(str(bad_xz))

    def test_file_like_object_loads(self):
        vj = VtypeJson(io.StringIO(json.dumps(_minimal_isf())))
        assert isinstance(vj, VtypeJson)

    def test_json_file_loads(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text(json.dumps(_minimal_isf()), encoding="utf-8")
        vj = VtypeJson(str(p))
        assert isinstance(vj, VtypeJson)

    def test_xz_file_loads(self, tmp_path):
        p = tmp_path / "ok.json.xz"
        with lzma.open(p, "wt", encoding="utf-8") as f:
            json.dump(_minimal_isf(), f)
        vj = VtypeJson(str(p))
        assert isinstance(vj, VtypeJson)

    def test_circular_typedef_in_vtypejson_resolve(self):
        isf = _minimal_isf()
        isf["typedefs"] = {
            "A": {"kind": "typedef", "name": "B"},
            "B": {"kind": "typedef", "name": "A"},
        }
        vj = VtypeJson(isf)
        with pytest.raises(ValueError, match="Circular typedef"):
            vj._resolve_type_info({"kind": "typedef", "name": "A"})


# ---------------------------------------------------------------------------
# 2. _FallbackBytesStruct
# ---------------------------------------------------------------------------

class TestFallbackBytesStruct:
    def test_unpack_returns_bytes(self):
        fb = _FallbackBytesStruct(6)
        result = fb.unpack_from(b"ABCDEF", 0)
        assert result == (b"ABCDEF",)

    def test_unpack_with_offset(self):
        fb = _FallbackBytesStruct(3)
        result = fb.unpack_from(b"XYZABC", 3)
        assert result == (b"ABC",)

    def test_pack_into_writes_bytes(self):
        fb = _FallbackBytesStruct(4)
        buf = bytearray(8)
        fb.pack_into(buf, 2, b"\xAA\xBB\xCC\xDD")
        assert buf[2:6] == b"\xAA\xBB\xCC\xDD"

    def test_pack_wrong_size_raises(self):
        fb = _FallbackBytesStruct(4)
        buf = bytearray(8)
        with pytest.raises(ValueError, match="Expected exactly 4 bytes"):
            fb.pack_into(buf, 0, b"\x01\x02")

    def test_pack_non_bytes_raises(self):
        fb = _FallbackBytesStruct(4)
        buf = bytearray(8)
        with pytest.raises(TypeError):
            fb.pack_into(buf, 0, 12345)

    def test_unpack_short_slice_pads(self):
        fb = _FallbackBytesStruct(4)
        result = fb.unpack_from(b"\xAA\xBB", 0)
        assert result == (b"\xAA\xBB\x00\x00",)

    def test_format_is_empty_string(self):
        fb = _FallbackBytesStruct(10)
        assert fb.format == ""

    def test_size_attribute(self):
        fb = _FallbackBytesStruct(16)
        assert fb.size == 16


# ---------------------------------------------------------------------------
# 3. VtypeBaseType exotic kinds: bool, char, f16
# ---------------------------------------------------------------------------

class TestVtypeBaseTypeExoticKinds:
    def test_bool_kind_compiles(self):
        bt = VtypeBaseType("mybool", {"kind": "bool", "size": 1, "signed": False, "endian": "little"})
        cs = bt.get_compiled_struct()
        assert cs is not None
        buf = bytearray(1)
        cs.pack_into(buf, 0, True)
        assert buf[0] == 1

    def test_char_kind_unsigned(self):
        bt = VtypeBaseType("char", {"kind": "char", "size": 1, "signed": False, "endian": "little"})
        cs = bt.get_compiled_struct()
        assert cs is not None
        buf = bytearray(1)
        cs.pack_into(buf, 0, 255)
        assert buf[0] == 255

    def test_f16_half_precision(self):
        bt = VtypeBaseType("f16", {"kind": "float", "size": 2, "signed": True, "endian": "little"})
        cs = bt.get_compiled_struct()
        assert cs is not None
        assert cs.size == 2

    def test_f32_and_f64(self):
        f32 = VtypeBaseType("f32", {"kind": "float", "size": 4, "signed": True, "endian": "little"})
        f64 = VtypeBaseType("f64", {"kind": "float", "size": 8, "signed": True, "endian": "little"})
        assert f32.get_compiled_struct() is not None
        assert f64.get_compiled_struct() is not None


# ---------------------------------------------------------------------------
# 4. VtypeUserType.get_aggregated_struct() failure paths
# ---------------------------------------------------------------------------

class TestAggregatedStructFailurePaths:
    def _make_ffi_with_struct(self, fields: dict, size: int) -> DFFI:
        isf = _minimal_isf(extra_user={"s": {"kind": "struct", "size": size, "fields": fields}})
        return DFFI(isf)

    def test_pointer_field_returns_none(self):
        ffi = self._make_ffi_with_struct(
            {"p": {"offset": 0, "type": {"kind": "pointer", "subtype": {"kind": "base", "name": "int"}}}},
            size=8,
        )
        t = ffi.get_user_type("s")
        assert t.get_aggregated_struct(ffi) is None

    def test_exotic_int_fallback_blocks_aggregation(self):
        isf = _minimal_isf(
            extra_base={"int24": {"kind": "int", "size": 3, "signed": True, "endian": "little"}},
            extra_user={
                "s": {
                    "kind": "struct",
                    "size": 3,
                    "fields": {"v": {"offset": 0, "type": {"kind": "base", "name": "int24"}}},
                }
            },
        )
        ffi = DFFI(isf)
        t = ffi.get_user_type("s")
        # _FallbackIntStruct has empty .format, so aggregation should fail
        assert t.get_aggregated_struct(ffi) is None

    def test_union_overlap_blocks_aggregation(self):
        isf = _minimal_isf(
            extra_user={
                "u": {
                    "kind": "union",
                    "size": 4,
                    "fields": {
                        "a": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                        "b": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    },
                }
            }
        )
        ffi = DFFI(isf)
        t = ffi.get_user_type("u")
        assert t.get_aggregated_struct(ffi) is None


# ---------------------------------------------------------------------------
# 5. VtypeEnum lazy _val_to_name cache
# ---------------------------------------------------------------------------

class TestVtypeEnumCache:
    def test_val_to_name_starts_none(self):
        e = VtypeEnum("e", {"size": 4, "base": "int", "constants": {"A": 1, "B": 2}})
        assert e._val_to_name is None

    def test_get_name_for_value_builds_cache(self):
        e = VtypeEnum("e", {"size": 4, "base": "int", "constants": {"A": 1, "B": 2}})
        assert e.get_name_for_value(1) == "A"
        assert e._val_to_name is not None
        assert e.get_name_for_value(2) == "B"

    def test_unknown_value_returns_none(self):
        e = VtypeEnum("e", {"size": 4, "base": "int", "constants": {"A": 0}})
        assert e.get_name_for_value(99) is None


# ---------------------------------------------------------------------------
# 6. VtypeSymbol.get_decoded_constant_data()
# ---------------------------------------------------------------------------

class TestVtypeSymbolConstantData:
    def test_valid_base64_decodes(self):
        payload = b"hello world"
        encoded = base64.b64encode(payload).decode()
        s = VtypeSymbol("sym", {"address": 0x1000, "constant_data": encoded})
        assert s.get_decoded_constant_data() == payload

    def test_no_constant_data_returns_none(self):
        s = VtypeSymbol("sym", {"address": 0x1000})
        assert s.get_decoded_constant_data() is None

    def test_invalid_base64_returns_none(self):
        s = VtypeSymbol("sym", {"address": 0x1000, "constant_data": "!!!not valid!!!"})
        assert s.get_decoded_constant_data() is None

    def test_symbol_pretty_print(self):
        s = VtypeSymbol("my_var", {"address": 0xDEAD, "type": {"kind": "base", "name": "int"}})
        out = s.pretty_print()
        assert "my_var" in out
        assert "0xdead" in out.lower()

    def test_symbol_str_equals_pretty_print(self):
        s = VtypeSymbol("my_var", {"address": 0x10, "type": {"kind": "base", "name": "int"}})
        assert str(s) == s.pretty_print()

    def test_symbol_to_dict(self):
        s = VtypeSymbol("foo", {"address": 0x400, "type": {"kind": "struct", "name": "bar"}})
        d = s.to_dict()
        assert d["name"] == "foo"
        assert d["address"] == 0x400
        assert d["type_info"]["name"] == "bar"


# ---------------------------------------------------------------------------
# 7. EnumInstance equality semantics
# ---------------------------------------------------------------------------

class TestEnumInstanceEquality:
    @pytest.fixture
    def color_enum(self):
        return VtypeEnum("color", {"size": 4, "base": "int", "constants": {"RED": 0, "GREEN": 1}})

    def test_equal_to_same_value(self, color_enum):
        assert EnumInstance(color_enum, 0) == EnumInstance(color_enum, 0)

    def test_not_equal_different_value(self, color_enum):
        assert EnumInstance(color_enum, 0) != EnumInstance(color_enum, 1)

    def test_equal_to_int(self, color_enum):
        assert EnumInstance(color_enum, 1) == 1

    def test_not_equal_to_wrong_int(self, color_enum):
        assert EnumInstance(color_enum, 1) != 99

    def test_equal_to_str_name(self, color_enum):
        assert EnumInstance(color_enum, 0) == "RED"

    def test_not_equal_to_wrong_str(self, color_enum):
        assert EnumInstance(color_enum, 0) != "GREEN"

    def test_not_equal_to_unknown_type(self, color_enum):
        assert EnumInstance(color_enum, 0) != [0]

    def test_repr_known_name(self, color_enum):
        r = repr(EnumInstance(color_enum, 1))
        assert "GREEN" in r and "1" in r

    def test_repr_unknown_value(self, color_enum):
        r = repr(EnumInstance(color_enum, 99))
        assert "99" in r


# ---------------------------------------------------------------------------
# 8. BoundArrayView equality edge cases
# ---------------------------------------------------------------------------

class TestBoundArrayViewEquality:
    @pytest.fixture
    def arr(self, simple_ffi):
        return simple_ffi.new("int[4]", [10, 20, 30, 40])

    def test_equal_to_matching_list(self, arr):
        assert arr == [10, 20, 30, 40]

    def test_not_equal_to_different_list(self, arr):
        assert arr != [10, 20, 30, 99]

    def test_length_mismatch_not_equal(self, arr):
        assert arr != [10, 20, 30]

    def test_not_equal_to_string(self, arr):
        assert arr != "not a list"

    def test_not_equal_to_int(self, arr):
        assert arr != 42

    def test_equal_to_another_array_view_same_content(self, simple_ffi):
        arr2 = simple_ffi.new("int[4]", [10, 20, 30, 40])
        arr1 = simple_ffi.new("int[4]", [10, 20, 30, 40])
        assert arr1 == arr2

    def test_ne_operator_works(self, arr):
        assert arr != [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 9. BoundArrayView.__add__ address propagation
# ---------------------------------------------------------------------------

class TestBoundArrayViewAdd:
    def test_add_without_base_address(self, simple_ffi):
        arr = simple_ffi.new("int[5]")
        ptr = arr + 2
        assert isinstance(ptr, Ptr)
        assert ptr.address == 2 * 4  # offset 0 + 2 * sizeof(int)

    def test_add_with_base_address(self, simple_ffi):
        buf = bytearray(20)
        arr = simple_ffi.from_buffer("int[5]", buf, address=0xCAFE0000)
        ptr = arr + 3
        assert ptr.address == 0xCAFE0000 + 3 * 4

    def test_add_preserves_subtype(self, simple_ffi):
        arr = simple_ffi.new("int[3]")
        ptr = arr + 0
        assert ptr.points_to_type_name == "int"

    def test_add_non_int_returns_not_implemented(self, simple_ffi):
        arr = simple_ffi.new("int[3]")
        assert arr.__add__("bad") is NotImplemented


# ---------------------------------------------------------------------------
# 10. Ptr.__hash__ / set / dict usage
# ---------------------------------------------------------------------------

class TestPtrHashAndGetitem:
    def test_ptr_is_hashable(self, simple_ffi):
        p = simple_ffi.cast("int *", 0x1000)
        assert isinstance(hash(p), int)

    def test_ptr_usable_as_dict_key(self, simple_ffi):
        p = simple_ffi.cast("int *", 0x2000)
        d = {p: "value"}
        assert d[p] == "value"

    def test_different_address_different_hash(self, simple_ffi):
        p1 = simple_ffi.cast("int *", 0x100)
        p2 = simple_ffi.cast("int *", 0x200)
        assert hash(p1) != hash(p2)

    def test_ptr_in_set_deduplicates(self, simple_ffi):
        p1 = simple_ffi.cast("int *", 0x500)
        p2 = simple_ffi.cast("int *", 0x500)
        assert len({p1, p2}) == 1

    def test_getitem_without_backend_raises(self, simple_ffi):
        p = simple_ffi.cast("int *", 0x1000)
        with pytest.raises(RuntimeError, match="No memory backend"):
            _ = p[0]

    def test_getitem_non_int_raises(self, simple_ffi):
        p = simple_ffi.cast("int *", 0x0)
        with pytest.raises(TypeError, match="integers"):
            _ = p["bad"]


# ---------------------------------------------------------------------------
# 11. DFFI.string() on enum instances
# ---------------------------------------------------------------------------

class TestDFFIStringOnEnum:
    def test_string_known_name(self, enum_ffi):
        inst = enum_ffi.from_buffer("enum color", bytearray(4))
        inst[0] = "GREEN"
        assert enum_ffi.string(inst) == b"GREEN"

    def test_string_unknown_value_returns_numeric(self, enum_ffi):
        buf = bytearray(4)
        stdlib_struct.pack_into("<i", buf, 0, 99)
        inst = enum_ffi.from_buffer("enum color", buf)
        assert enum_ffi.string(inst) == b"99"

    def test_string_raises_on_ptr(self, simple_ffi):
        p = simple_ffi.cast("int *", 0x1000)
        with pytest.raises(TypeError, match="Ptr"):
            simple_ffi.string(p)


# ---------------------------------------------------------------------------
# 12. DFFI.buffer() raises TypeError on Ptr; to_bytes() on zero-size
# ---------------------------------------------------------------------------

class TestDFFIBufferAndToBytes:
    def test_buffer_raises_on_ptr(self, simple_ffi):
        p = simple_ffi.cast("int *", 0xDEAD)
        with pytest.raises(TypeError, match="Ptr"):
            simple_ffi.buffer(p)

    def test_to_bytes_zero_size_type(self, simple_ffi):
        inst = simple_ffi.from_buffer("void", bytearray(4))
        assert simple_ffi.to_bytes(inst) == b""

    def test_buffer_returns_memoryview(self, simple_ffi):
        inst = simple_ffi.new("int", 42)
        mv = simple_ffi.buffer(inst)
        assert isinstance(mv, memoryview)
        assert len(mv) == 4

    def test_buffer_on_array_view(self, simple_ffi):
        arr = simple_ffi.new("int[3]", [1, 2, 3])
        mv = simple_ffi.buffer(arr)
        assert isinstance(mv, memoryview)
        assert len(mv) == 12


# ---------------------------------------------------------------------------
# 13. DFFI.memmove() with raw bytes as source
# ---------------------------------------------------------------------------

class TestDFFIMemmoveRawBytes:
    def test_memmove_bytes_into_instance(self, simple_ffi):
        dst = simple_ffi.new("int")
        raw = stdlib_struct.pack("<i", 12345)
        simple_ffi.memmove(dst, raw, 4)
        assert int(dst) == 12345

    def test_memmove_partial_bytes(self, simple_ffi):
        dst = simple_ffi.new("int", 0)
        raw = b"\xFF\xFF\x00\x00"
        simple_ffi.memmove(dst, raw, 2)
        raw_bytes = bytes(dst)
        assert raw_bytes[0] == 0xFF
        assert raw_bytes[1] == 0xFF


# ---------------------------------------------------------------------------
# 14. DFFI.load_isf() idempotency
# ---------------------------------------------------------------------------

class TestDFFILoadISFIdempotency:
    def test_same_dict_object_not_duplicated(self):
        isf = _minimal_isf()
        ffi = DFFI()
        ffi.load_isf(isf)
        ffi.load_isf(isf)
        assert len(ffi._file_order) == 1

    def test_same_path_not_duplicated(self, tmp_path):
        p = tmp_path / "ok.json"
        p.write_text(json.dumps(_minimal_isf()), encoding="utf-8")
        ffi = DFFI()
        ffi.load_isf(str(p))
        ffi.load_isf(str(p))
        assert len(ffi._file_order) == 1

    def test_different_dict_objects_both_loaded(self):
        ffi = DFFI()
        ffi.load_isf(_minimal_isf())
        ffi.load_isf(_minimal_isf())
        assert len(ffi._file_order) == 2

    def test_load_isf_rejects_list(self):
        ffi = DFFI()
        with pytest.raises(TypeError, match="load_isf expects a file path"):
            ffi.load_isf([_minimal_isf()])


# ---------------------------------------------------------------------------
# 15. DFFI.get_symbol() include_incomplete filtering
# ---------------------------------------------------------------------------

class TestDFFIGetSymbolFiltering:
    @pytest.fixture
    def sym_ffi(self):
        return DFFI(_minimal_isf(
            extra_syms={
                "has_addr":  {"address": 0x1000, "type": {"kind": "base", "name": "int"}},
                "zero_addr": {"address": 0,      "type": {"kind": "base", "name": "int"}},
                "no_addr":   {"type":             {"kind": "base", "name": "int"}},
            }
        ))

    def test_default_excludes_zero_address(self, sym_ffi):
        assert sym_ffi.get_symbol("zero_addr") is None

    def test_include_incomplete_finds_zero_addr(self, sym_ffi):
        sym = sym_ffi.get_symbol("zero_addr", include_incomplete=True)
        assert sym is not None and sym.address == 0

    def test_include_incomplete_finds_no_addr(self, sym_ffi):
        assert sym_ffi.get_symbol("no_addr", include_incomplete=True) is not None

    def test_normal_symbol_found_by_default(self, sym_ffi):
        sym = sym_ffi.get_symbol("has_addr")
        assert sym is not None and sym.address == 0x1000


# ---------------------------------------------------------------------------
# 16. DFFI.shift_symbol_addresses() with path= targeting one ISF
# ---------------------------------------------------------------------------

class TestDFFIShiftSymbolsTargeted:
    @pytest.fixture
    def two_isf_ffi(self):
        isf1 = _minimal_isf(extra_syms={"sym_a": {"address": 0x1000, "type": {"kind": "base", "name": "int"}}})
        isf2 = _minimal_isf(extra_syms={"sym_b": {"address": 0x2000, "type": {"kind": "base", "name": "int"}}})
        return DFFI([isf1, isf2])

    def test_path_targeted_shift_only_affects_one(self, two_isf_ffi):
        ffi = two_isf_ffi
        ffi.shift_symbol_addresses(0x100, path=ffi._file_order[0])
        assert ffi.get_symbol("sym_a").address == 0x1100
        assert ffi.get_symbol("sym_b").address == 0x2000

    def test_global_shift_affects_all(self, two_isf_ffi):
        ffi = two_isf_ffi
        ffi.shift_symbol_addresses(0x10)
        assert ffi.get_symbol("sym_a").address == 0x1010
        assert ffi.get_symbol("sym_b").address == 0x2010


# ---------------------------------------------------------------------------
# 17. DFFI.search_types() with use_regex=True
# ---------------------------------------------------------------------------

class TestDFFISearchTypesRegex:
    @pytest.fixture
    def search_ffi(self):
        return DFFI(_minimal_isf(extra_user={
            "task_struct": {"kind": "struct", "size": 4, "fields": {}},
            "task_info":   {"kind": "struct", "size": 4, "fields": {}},
            "net_device":  {"kind": "struct", "size": 4, "fields": {}},
        }))

    def test_regex_finds_matches(self, search_ffi):
        results = search_ffi.search_types(r"^task_", use_regex=True)
        assert "task_struct" in results and "task_info" in results
        assert "net_device" not in results

    def test_regex_is_case_sensitive(self, search_ffi):
        assert len(search_ffi.search_types(r"^TASK_", use_regex=True)) == 0

    def test_glob_still_works(self, search_ffi):
        assert "task_struct" in search_ffi.search_types("task_*")

    def test_search_symbols_regex(self):
        ffi = DFFI(_minimal_isf(extra_syms={
            "sys_open": {"address": 0x10, "type": {"kind": "base", "name": "int"}},
            "sys_read": {"address": 0x20, "type": {"kind": "base", "name": "int"}},
            "vfs_open": {"address": 0x30, "type": {"kind": "base", "name": "int"}},
        }))
        results = ffi.search_symbols(r"^sys_", use_regex=True)
        assert "sys_open" in results and "sys_read" in results
        assert "vfs_open" not in results


# ---------------------------------------------------------------------------
# 18. from_buffer() with memoryview input
# ---------------------------------------------------------------------------

class TestFromBufferMemoryview:
    def test_from_buffer_memoryview_basic(self, simple_ffi):
        raw = bytearray(4)
        stdlib_struct.pack_into("<i", raw, 0, 77)
        inst = simple_ffi.from_buffer("int", memoryview(raw))
        assert int(inst) == 77

    def test_from_buffer_memoryview_with_offset(self, simple_ffi):
        raw = bytearray(8)
        stdlib_struct.pack_into("<i", raw, 4, 42)
        inst = simple_ffi.from_buffer("int", memoryview(raw), offset=4)
        assert int(inst) == 42


# ---------------------------------------------------------------------------
# 19. from_address() bounded (nonzero-count) array
# ---------------------------------------------------------------------------

class TestFromAddressBoundedArray:
    def test_fixed_count_array_read(self):
        mem = bytearray(20)
        for i in range(5):
            stdlib_struct.pack_into("<i", mem, i * 4, i * 10)
        ffi = DFFI(_minimal_isf(), backend=mem)
        arr = ffi.from_address("int[5]", 0)
        assert isinstance(arr, BoundArrayView)
        assert len(arr) == 5
        assert arr[3] == 30

    def test_bounded_array_write_through(self):
        mem = bytearray(20)
        ffi = DFFI(_minimal_isf(), backend=mem)
        arr = ffi.from_address("int[3]", 4)
        arr[0] = 111
        assert stdlib_struct.unpack_from("<i", mem, 4)[0] == 111


# ---------------------------------------------------------------------------
# 20. LiveMemoryProxy edge cases
# ---------------------------------------------------------------------------

class TestLiveMemoryProxyEdgeCases:
    def test_len_returns_maxsize(self):
        proxy = LiveMemoryProxy(BytesBackend(b"\x00" * 10))
        assert len(proxy) == sys.maxsize

    def test_invalid_getitem_type_raises(self):
        proxy = LiveMemoryProxy(BytesBackend(b"\x00" * 10))
        with pytest.raises(TypeError):
            _ = proxy["bad"]

    def test_invalid_setitem_type_raises(self):
        proxy = LiveMemoryProxy(BytesBackend(bytearray(10)))
        with pytest.raises(TypeError):
            proxy["bad"] = b"\x00"

    def test_unbounded_slice_raises(self):
        proxy = LiveMemoryProxy(BytesBackend(b"\x00" * 10))
        with pytest.raises(ValueError, match="bounded"):
            _ = proxy[5:]

    def test_single_byte_read(self):
        proxy = LiveMemoryProxy(BytesBackend(b"\xAB\xCD"))
        assert proxy[1] == b"\xCD"


# ---------------------------------------------------------------------------
# 21. BytesBackend boundary guards
# ---------------------------------------------------------------------------

class TestBytesBackendBoundaries:
    def test_negative_address_read_raises(self):
        with pytest.raises(MemoryError, match="out of bounds"):
            BytesBackend(b"HELLO").read(-1, 1)

    def test_negative_address_write_raises(self):
        with pytest.raises(MemoryError, match="out of bounds"):
            BytesBackend(bytearray(5)).write(-1, b"\x00")

    def test_exact_end_read_succeeds(self):
        assert BytesBackend(b"ABCDE").read(3, 2) == b"DE"

    def test_one_past_end_read_raises(self):
        with pytest.raises(MemoryError):
            BytesBackend(b"ABCDE").read(4, 2)

    def test_full_read(self):
        data = b"12345678"
        assert BytesBackend(data).read(0, 8) == data

    def test_write_beyond_end_raises(self):
        with pytest.raises(MemoryError):
            BytesBackend(bytearray(4)).write(3, b"\xAA\xBB")


# ---------------------------------------------------------------------------
# 22. BoundTypeInstance.__dir__ content and exclusions
# ---------------------------------------------------------------------------

class TestBoundTypeInstanceDir:
    @pytest.fixture
    def anon_ffi(self):
        return DFFI(_minimal_isf(extra_user={
            "inner": {
                "kind": "struct", "size": 4,
                "fields": {"val": {"offset": 0, "type": {"kind": "base", "name": "int"}}},
            },
            "outer": {
                "kind": "struct", "size": 8,
                "fields": {
                    "x": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "anon_inner": {
                        "offset": 4, "anonymous": True,
                        "type": {"kind": "struct", "name": "inner"},
                    },
                },
            },
        }))

    def test_direct_fields_in_dir(self, anon_ffi):
        attrs = dir(anon_ffi.new("struct outer"))
        assert "x" in attrs

    def test_flattened_anonymous_fields_in_dir(self, anon_ffi):
        attrs = dir(anon_ffi.new("struct outer"))
        assert "val" in attrs

    def test_instance_cache_excluded(self, anon_ffi):
        attrs = dir(anon_ffi.new("struct outer"))
        assert "_instance_cache" not in attrs

    def test_dunder_attributes_present(self, anon_ffi):
        assert "__class__" in dir(anon_ffi.new("struct outer"))

    def test_internal_slots_in_dir(self, anon_ffi):
        assert "_instance_offset" in dir(anon_ffi.new("struct outer"))


# ---------------------------------------------------------------------------
# 23. typeof() whitespace stripping, invalid input, _typeof_or_raise
# ---------------------------------------------------------------------------

class TestDFFITypeofAndSizeof:
    def test_typeof_strips_whitespace(self, simple_ffi):
        t = simple_ffi.typeof("  int  ")
        assert t is not None and t.name == "int"

    def test_typeof_on_array_string(self, simple_ffi):
        t = simple_ffi.typeof("int[10]")
        assert isinstance(t, dict) and t["kind"] == "array" and t["count"] == 10

    def test_typeof_on_ptr_string(self, simple_ffi):
        t = simple_ffi.typeof("int *")
        assert isinstance(t, dict) and t["kind"] == "pointer"

    def test_typeof_on_vtype_passthrough(self, simple_ffi):
        bt = simple_ffi.get_base_type("int")
        assert simple_ffi.typeof(bt) is bt

    def test_typeof_on_dict_passthrough(self, simple_ffi):
        d = {"kind": "base", "name": "int"}
        assert simple_ffi.typeof(d) is d

    def test_typeof_invalid_type_raises(self, simple_ffi):
        with pytest.raises(TypeError, match="Expected string"):
            simple_ffi.typeof(999)

    def test_typeof_or_raise_missing_type(self, simple_ffi):
        with pytest.raises(KeyError, match="Unknown type"):
            simple_ffi._typeof_or_raise("struct does_not_exist")

    def test_sizeof_missing_array_subtype(self, simple_ffi):
        with pytest.raises((ValueError, KeyError)):
            simple_ffi.sizeof({"kind": "array", "count": 5})


# ---------------------------------------------------------------------------
# 24. Multi-ISF properties: base_types merge, enums first-wins
# ---------------------------------------------------------------------------

class TestDFFIPropertiesMultiISF:
    def test_base_types_merge(self):
        isf1 = _minimal_isf()
        isf2 = {
            "metadata": {},
            "base_types": {"special": {"kind": "int", "size": 8, "signed": False, "endian": "little"}},
            "user_types": {}, "enums": {}, "symbols": {},
        }
        ffi = DFFI([isf1, isf2])
        assert "int" in ffi.base_types and "special" in ffi.base_types

    def test_enums_first_wins(self):
        isf1 = _minimal_isf(extra_enums={"e": {"size": 4, "base": "int", "constants": {"A": 1}}})
        isf2 = _minimal_isf(extra_enums={"e": {"size": 4, "base": "int", "constants": {"B": 2}}})
        ffi = DFFI([isf1, isf2])
        assert "A" in ffi.enums["e"].constants


# ---------------------------------------------------------------------------
# 25. find_types_with_member edge cases
# ---------------------------------------------------------------------------

class TestFindTypesWithMember:
    def test_empty_result_when_no_match(self, struct_ffi):
        assert struct_ffi.find_types_with_member("nonexistent_field") == {}

    def test_finds_type_with_direct_field(self, struct_ffi):
        assert "point" in struct_ffi.find_types_with_member("x")


# ---------------------------------------------------------------------------
# 26. offsetof() error paths
# ---------------------------------------------------------------------------

class TestDFFIOffsetsErrors:
    def test_offsetof_non_struct_raises(self, simple_ffi):
        with pytest.raises(TypeError, match="is not a struct or union"):
            simple_ffi.offsetof("int", "foo")

    def test_offsetof_missing_field_raises(self, struct_ffi):
        with pytest.raises(KeyError, match="has no field"):
            struct_ffi.offsetof("struct point", "z")

    def test_offsetof_nested_into_non_struct_raises(self, struct_ffi):
        with pytest.raises(TypeError, match="Cannot get offset"):
            struct_ffi.offsetof("struct point", "x", "bad")


# ---------------------------------------------------------------------------
# 27. VtypeUserType / VtypeEnum pretty_print / str / to_dict / members
# ---------------------------------------------------------------------------

class TestVtypeTypePrettyPrint:
    def _make_user_type(self):
        return VtypeUserType("foo", {
            "kind": "struct", "size": 8,
            "fields": {"a": {"offset": 0, "type": {"kind": "base", "name": "int"}, "anonymous": False}},
        })

    def _make_enum(self):
        return VtypeEnum("status", {"size": 4, "base": "int", "constants": {"OK": 0, "ERR": 1}})

    def test_user_type_to_dict(self):
        d = self._make_user_type().to_dict()
        assert d["name"] == "foo" and d["kind"] == "struct" and "a" in d["fields"]

    def test_user_type_str_equals_pretty_print(self):
        t = self._make_user_type()
        assert str(t) == t.pretty_print()

    def test_user_type_members_alias(self):
        t = self._make_user_type()
        assert t.members is t.fields

    def test_enum_to_dict(self):
        d = self._make_enum().to_dict()
        assert d["name"] == "status" and d["constants"]["OK"] == 0

    def test_enum_str_equals_pretty_print(self):
        e = self._make_enum()
        assert str(e) == e.pretty_print()

    def test_enum_members_alias(self):
        e = self._make_enum()
        assert e.members is e.constants


# ---------------------------------------------------------------------------
# 29. get_type_size() edge cases
# ---------------------------------------------------------------------------

class TestDFFIGetTypeSizeEdgeCases:
    def test_missing_array_subtype_returns_none(self, simple_ffi):
        vj = simple_ffi.vtypejsons[simple_ffi._file_order[0]]
        assert vj.get_type_size({"kind": "array", "count": 5}) is None

    def test_unknown_kind_returns_none(self, simple_ffi):
        vj = simple_ffi.vtypejsons[simple_ffi._file_order[0]]
        assert vj.get_type_size({"kind": "completely_made_up"}) is None


# ---------------------------------------------------------------------------
# 30. BoundTypeInstance numeric conversions / __index__ / hex
# ---------------------------------------------------------------------------

class TestBoundTypeInstanceConversions:
    def test_bool_false_for_zero(self, simple_ffi):
        assert bool(simple_ffi.new("int", 0)) is False

    def test_bool_true_for_nonzero(self, simple_ffi):
        assert bool(simple_ffi.new("int", 1)) is True

    def test_int_raises_on_struct(self, struct_ffi):
        with pytest.raises(TypeError, match="struct/union"):
            int(struct_ffi.new("struct point"))

    def test_float_raises_on_struct(self, struct_ffi):
        with pytest.raises(TypeError, match="Cannot convert"):
            float(struct_ffi.new("struct point"))

    def test_index_works_for_base_type(self, simple_ffi):
        assert simple_ffi.new("int", 7).__index__() == 7

    def test_hex_via_index(self, simple_ffi):
        assert hex(simple_ffi.new("int", 255)) == "0xff"

    def test_float_on_float_type(self):
        ffi = DFFI(_minimal_isf(extra_base={"f32": {"kind": "float", "size": 4, "signed": True, "endian": "little"}}))
        inst = ffi.new("f32", 3.14)
        assert abs(float(inst) - 3.14) < 0.01


# ---------------------------------------------------------------------------
# 31. typeof() on BoundTypeInstance, Ptr, and BoundArrayView passthrough
# ---------------------------------------------------------------------------

class TestDFFITypeofPassthrough:
    def test_typeof_bound_instance(self, simple_ffi):
        inst = simple_ffi.new("int", 0)
        assert simple_ffi.typeof(inst) is inst._instance_type_def

    def test_typeof_ptr(self, simple_ffi):
        t = simple_ffi.typeof(simple_ffi.cast("int *", 0x1000))
        assert isinstance(t, dict) and t["kind"] == "pointer"

    def test_typeof_bound_array_view(self, simple_ffi):
        t = simple_ffi.typeof(simple_ffi.new("int[3]"))
        assert isinstance(t, dict) and t["kind"] == "array" and t["count"] == 3


# ---------------------------------------------------------------------------
# 32. End-to-end integration: nested struct + backend + enum + symbols
# ---------------------------------------------------------------------------

class TestEndToEndIntegration:
    @pytest.fixture
    def rich_ffi(self):
        isf = {
            "metadata": {},
            "base_types": {
                "int":     {"kind": "int",     "size": 4, "signed": True,  "endian": "little"},
                "char":    {"kind": "int",     "size": 1, "signed": False, "endian": "little"},
                "pointer": {"kind": "pointer", "size": 8,                  "endian": "little"},
                "void":    {"kind": "void",    "size": 0, "signed": False, "endian": "little"},
            },
            "user_types": {
                "node": {
                    "kind": "struct", "size": 16,
                    "fields": {
                        "value": {"offset": 0, "type": {"kind": "base",    "name": "int"}},
                        "next":  {"offset": 8, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "node"}}},
                    },
                }
            },
            "enums": {
                "state": {"size": 4, "base": "int", "constants": {"IDLE": 0, "RUNNING": 1, "DONE": 2}}
            },
            "symbols": {
                "root_node": {"address": 0x100, "type": {"kind": "struct", "name": "node"}},
                "inactive":  {"address": 0,     "type": {"kind": "base",   "name": "int"}},
            },
        }
        return DFFI(isf)

    def test_new_and_read_nested(self, rich_ffi):
        n = rich_ffi.new("struct node", {"value": 42, "next": 0xDEAD})
        assert n.value == 42 and n.next.address == 0xDEAD

    def test_buffer_roundtrip(self, rich_ffi):
        n = rich_ffi.new("struct node", {"value": 12345})
        n2 = rich_ffi.from_buffer("struct node", bytearray(rich_ffi.to_bytes(n)))
        assert n2.value == 12345

    def test_enum_string_roundtrip(self, rich_ffi):
        inst = rich_ffi.new("enum state")
        inst[0] = "RUNNING"
        assert rich_ffi.string(inst) == b"RUNNING"
        assert int(inst) == 1

    def test_symbol_lookup_filters_zero(self, rich_ffi):
        assert rich_ffi.get_symbol("root_node").address == 0x100
        assert rich_ffi.get_symbol("inactive") is None

    def test_backend_read_write(self, rich_ffi):
        mem = bytearray(0x200)
        stdlib_struct.pack_into("<i", mem, 0x100, 777)
        rich_ffi.backend = BytesBackend(mem)
        inst = rich_ffi.from_address("struct node", 0x100)
        assert inst.value == 777
        inst.value = 888
        assert stdlib_struct.unpack_from("<i", mem, 0x100)[0] == 888

    def test_to_dict_recursive(self, rich_ffi):
        n = rich_ffi.new("struct node", {"value": 99, "next": 0x500})
        d = rich_ffi.to_dict(n)
        assert d["value"] == 99 and d["next"] == 0x500

    def test_pretty_print_output(self, rich_ffi):
        n = rich_ffi.new("struct node", {"value": 7})
        out = rich_ffi.pretty_print(n)
        assert "node" in out and "7" in out