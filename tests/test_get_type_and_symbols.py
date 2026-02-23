from __future__ import annotations

from dwarffi.core import VtypeJson, isf_from_dict


def test_get_type_prefixes(base_types_little_endian) -> None:
    isf = isf_from_dict(
        {
            "metadata": {},
            "base_types": base_types_little_endian,
            "user_types": {
                "my_struct": {"kind": "struct", "size": 4, "fields": {}},
                "my_union": {"kind": "union", "size": 4, "fields": {}},
            },
            "enums": {"my_enum": {"size": 4, "base": "int", "constants": {"A": 1}}},
            "symbols": {},
        }
    )

    assert isf.get_type("my_struct") is not None
    assert isf.get_type("struct my_struct") is not None
    assert isf.get_type("union my_union") is not None
    assert isf.get_type("enum my_enum") is not None


def test_get_symbols_by_address_and_shift(base_types_little_endian) -> None:
    target = 0x1000
    delta = 0x10

    isf = isf_from_dict(
        {
            "metadata": {},
            "base_types": base_types_little_endian,
            "user_types": {},
            "enums": {},
            "symbols": {
                "sym1": {"address": target},
                "sym2": {"address": target},
                "sym3": {"address": target + 1},
            },
        }
    )

    # Prime symbol cache to ensure shift updates cached objects too.
    sym1 = isf.get_symbol("sym1")
    assert sym1 is not None
    assert sym1.address == target

    at_target = isf.get_symbols_by_address(target)
    assert {s.name for s in at_target} == {"sym1", "sym2"}

    isf.shift_symbol_addresses(delta)

    assert sym1.address == target + delta
    assert isf.get_symbols_by_address(target) == []
    at_shifted = isf.get_symbols_by_address(target + delta)
    assert {s.name for s in at_shifted} == {"sym1", "sym2"}


def test_vtypejson_repr_smoke(base_types_little_endian) -> None:
    isf = isf_from_dict(
        {
            "metadata": {},
            "base_types": base_types_little_endian,
            "user_types": {},
            "enums": {},
            "symbols": {},
        }
    )
    assert isinstance(repr(isf), str)
    assert isinstance(isf, VtypeJson)
