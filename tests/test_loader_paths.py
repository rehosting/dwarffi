from __future__ import annotations

import io
import json
import lzma
import pytest

from dwarffi import VtypeJson, load_isf_json
from dwarffi import DFFI


def test_load_isf_json_supports_path_and_xz(tmp_path) -> None:
    isf_dict = {
        "metadata": {},
        "base_types": {"int": {"size": 4, "signed": True, "kind": "int", "endian": "little"}},
        "user_types": {},
        "enums": {},
        "symbols": {},
    }

    json_path = tmp_path / "sample.isf.json"
    json_path.write_text(json.dumps(isf_dict), encoding="utf-8")
    isf = load_isf_json(str(json_path))
    assert isinstance(isf, VtypeJson)

    xz_path = tmp_path / "sample.isf.json.xz"
    with lzma.open(xz_path, "wt", encoding="utf-8") as f:
        json.dump(isf_dict, f)
    isf_xz = load_isf_json(str(xz_path))
    assert isinstance(isf_xz, VtypeJson)


def test_load_isf_json_supports_file_like() -> None:
    isf_dict = {
        "metadata": {},
        "base_types": {"int": {"size": 4, "signed": True, "kind": "int", "endian": "little"}},
        "user_types": {},
        "enums": {},
        "symbols": {},
    }
    f = io.StringIO(json.dumps(isf_dict))
    isf = load_isf_json(f)
    assert isinstance(isf, VtypeJson)

def test_dffi_init_supports_list_but_load_isf_is_singular(tmp_path):
    dict_a = {"metadata": {}, "base_types": {"int": {"size": 4}}, "user_types": {}, "enums": {}, "symbols": {}}
    dict_b = {"metadata": {}, "base_types": {"char": {"size": 1}}, "user_types": {}, "enums": {}, "symbols": {}}

    # 1. Constructor should support the list
    ffi = DFFI([dict_a, dict_b])
    assert ffi.sizeof("int") == 4
    assert ffi.sizeof("char") == 1
    assert len(ffi._file_order) == 2

    # 2. load_isf should raise TypeError if passed a list directly
    ffi_singular = DFFI()
    with pytest.raises(TypeError, match="load_isf expects a file path"):
        ffi_singular.load_isf([dict_a, dict_b])
    
    # 3. load_isf should work for singular dict
    ffi_singular.load_isf(dict_a)
    assert ffi_singular.sizeof("int") == 4