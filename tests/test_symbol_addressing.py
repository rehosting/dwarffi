import pytest

from dwarffi import DFFI


def _isf_with_symbols(*, endian: str, ptr_size: int, symbols: dict) -> dict:
    # Minimal ISF dict for VtypeJson + symbol behavior
    return {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": endian},
            "pointer": {"kind": "pointer", "size": ptr_size, "endian": endian},
            "void": {"kind": "void", "size": 0, "signed": False, "endian": endian},
        },
        "user_types": {},
        "enums": {},
        # symbols: { name: { "address": int, "type": {...} } }
        "symbols": symbols,
    }


def test_get_symbols_by_address_aggregates_across_isfs():
    isf1 = _isf_with_symbols(
        endian="little",
        ptr_size=8,
        symbols={
            "sym_a": {"address": 0x1000, "type": {"kind": "base", "name": "int"}},
        },
    )
    isf2 = _isf_with_symbols(
        endian="little",
        ptr_size=8,
        symbols={
            "sym_b": {"address": 0x1000, "type": {"kind": "base", "name": "int"}},
        },
    )

    ffi = DFFI([isf1, isf2])

    syms = ffi.get_symbols_by_address(0x1000)
    assert sorted(s.name for s in syms) == ["sym_a", "sym_b"]


def test_shift_symbol_addresses_path_only_shifts_target_isf_and_invalidates_reverse_cache():
    isf1 = _isf_with_symbols(
        endian="little",
        ptr_size=8,
        symbols={
            "sym_a": {"address": 0x1000, "type": {"kind": "base", "name": "int"}},
        },
    )
    isf2 = _isf_with_symbols(
        endian="little",
        ptr_size=8,
        symbols={
            "sym_b": {"address": 0x1000, "type": {"kind": "base", "name": "int"}},
        },
    )

    ffi = DFFI([isf1, isf2])
    path1, path2 = ffi._file_order  # pseudo-paths like <dict_...>

    # Prime reverse-lookup caches in BOTH VtypeJson instances
    assert sorted(s.name for s in ffi.get_symbols_by_address(0x1000)) == ["sym_a", "sym_b"]

    # Shift ONLY the first ISF
    ffi.shift_symbol_addresses(0x100, path=path1)

    # Old address should now only contain sym_b (since sym_a moved)
    assert [s.name for s in ffi.get_symbols_by_address(0x1000)] == ["sym_b"]

    # New address should contain only sym_a
    assert [s.name for s in ffi.get_symbols_by_address(0x1100)] == ["sym_a"]

    # Sanity: second ISF unchanged
    assert ffi.get_symbol("sym_b").address == 0x1000
    assert ffi.get_symbol("sym_a").address == 0x1100


def test_shift_symbol_addresses_global_shifts_all_isfs():
    isf1 = _isf_with_symbols(
        endian="little",
        ptr_size=8,
        symbols={"sym_a": {"address": 0x2000, "type": {"kind": "base", "name": "int"}}},
    )
    isf2 = _isf_with_symbols(
        endian="little",
        ptr_size=8,
        symbols={"sym_b": {"address": 0x3000, "type": {"kind": "base", "name": "int"}}},
    )

    ffi = DFFI([isf1, isf2])

    ffi.shift_symbol_addresses(0x10)

    assert ffi.get_symbol("sym_a").address == 0x2010
    assert ffi.get_symbol("sym_b").address == 0x3010