import pytest
from dwarffi import DFFI

@pytest.fixture
def ui_ffi():
    """Provides a DFFI instance with a mix of types and symbols to test the UI."""
    isf = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"}
        },
        "user_types": {
            "my_struct": {
                "kind": "struct",
                "size": 8,
                "fields": {
                    "a": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "__anon_field": {"offset": 4, "anonymous": True, "type": {"kind": "base", "name": "int"}}
                }
            }
        },
        "enums": {
            "my_state": {
                "size": 4,
                "base": "int",
                "constants": {"IDLE": 0, "RUNNING": 1}
            }
        },
        "symbols": {
            "sys_call_table": {"address": 0xffffffff82000280, "type": {"kind": "array", "subtype": {"name": "sys_call_ptr_t"}}},
            "null_symbol": {"address": 0x0, "type": {"kind": "base", "name": "int"}}
        },
        "typedefs": {}
    }
    return DFFI(isf)

def test_dffi_global_properties(ui_ffi):
    """Test that DFFI exposes the global dictionary properties correctly."""
    # 1. Base Types
    assert "int" in ui_ffi.base_types
    assert ui_ffi.base_types["int"].size == 4

    # 2. User Types
    assert "my_struct" in ui_ffi.types
    assert ui_ffi.types["my_struct"].kind == "struct"

    # 3. Enums
    assert "my_state" in ui_ffi.enums
    assert ui_ffi.enums["my_state"].size == 4

    # 4. Symbols
    assert "sys_call_table" in ui_ffi.symbols
    assert "null_symbol" in ui_ffi.symbols
    assert ui_ffi.symbols["sys_call_table"].address == 0xffffffff82000280

def test_vtype_user_type_ui(ui_ffi):
    """Test the UI introspection methods on VtypeUserType (Structs/Unions)."""
    t = ui_ffi.types["my_struct"]
    
    # .members alias
    assert "a" in t.members
    assert "__anon_field" in t.members
    assert t.members is t.fields

    # .to_dict()
    d = t.to_dict()
    assert d["name"] == "my_struct"
    assert d["kind"] == "struct"
    assert d["size"] == 8
    assert d["fields"]["a"]["offset"] == 0
    assert d["fields"]["__anon_field"]["anonymous"] is True

    # .pretty_print() and str()
    out = t.pretty_print()
    assert str(t) == out
    assert "struct my_struct (size: 8 bytes) {" in out
    assert "[+0  ] base int a;" in out
    assert "[+4  ] <anonymous> base int;" in out

def test_vtype_enum_ui(ui_ffi):
    """Test the UI introspection methods on VtypeEnum."""
    e = ui_ffi.enums["my_state"]
    
    # .members alias
    assert "IDLE" in e.members
    assert e.members["RUNNING"] == 1
    assert e.members is e.constants

    # .to_dict()
    d = e.to_dict()
    assert d["name"] == "my_state"
    assert d["size"] == 4
    assert d["base"] == "int"
    assert d["constants"]["IDLE"] == 0

    # .pretty_print() and str()
    out = e.pretty_print()
    assert str(e) == out
    assert "enum my_state (size: 4, base: int) {" in out
    assert "IDLE = 0," in out
    assert "RUNNING = 1," in out

def test_vtype_symbol_ui(ui_ffi):
    """Test the UI introspection methods on VtypeSymbol."""
    s = ui_ffi.symbols["sys_call_table"]
    
    # .to_dict()
    d = s.to_dict()
    assert d["name"] == "sys_call_table"
    assert d["address"] == 0xffffffff82000280
    assert d["type_info"]["kind"] == "array"

    # .pretty_print() and str()
    out = s.pretty_print()
    assert str(s) == out
    assert "Symbol sys_call_table @ 0xffffffff82000280" in out
    assert "Type: array sys_call_ptr_t" in out

def test_multi_isf_first_wins_resolution(tmp_path):
    """Ensures the properties correctly merge multiple ISFs using first-wins logic."""
    isf_core = {
        "metadata": {}, "base_types": {}, "user_types": {}, "enums": {},
        "symbols": {
            "shared_sym": {"address": 0x1000, "type": {"kind": "base", "name": "int"}},
            "core_only": {"address": 0x2000, "type": {"kind": "base", "name": "int"}}
        }
    }
    isf_plugin = {
        "metadata": {}, "base_types": {}, "user_types": {}, "enums": {},
        "symbols": {
            "shared_sym": {"address": 0x9999, "type": {"kind": "base", "name": "int"}},
            "plugin_only": {"address": 0x3000, "type": {"kind": "base", "name": "int"}}
        }
    }
    
    ffi = DFFI([isf_core, isf_plugin])
    syms = ffi.symbols
    
    # Should contain symbols from both
    assert "core_only" in syms
    assert "plugin_only" in syms
    
    # Shared symbol should resolve to the FIRST loaded file (core)
    assert syms["shared_sym"].address == 0x1000