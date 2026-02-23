from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def base_types_little_endian() -> dict[str, Any]:
    return {
        "int": {"size": 4, "signed": True, "kind": "int", "endian": "little"},
        "u8": {"size": 1, "signed": False, "kind": "int", "endian": "little"},
        "u64": {"size": 8, "signed": False, "kind": "int", "endian": "little"},
        "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"},
        "void": {"size": 0, "signed": False, "kind": "void", "endian": "little"},
    }
