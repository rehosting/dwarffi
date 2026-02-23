from __future__ import annotations

import io
import json
import lzma

from dwarffi.core import VtypeJson, load_isf_json


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
