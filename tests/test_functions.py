import pytest

from dwarffi import DFFI


@pytest.fixture
def mock_isf_with_functions():
    """A mock ISF dictionary containing the new custom functions schema."""
    return {
        "metadata": {"format": "6.2.0", "producer": {"name": "test", "version": "1.0"}},
        "base_types": {
            "int": {"size": 4, "signed": True, "kind": "int", "endian": "little"},
            "void": {"size": 0, "signed": False, "kind": "void", "endian": "little"}
        },
        "user_types": {},
        "enums": {},
        "symbols": {},
        "functions": {
            "sys_read": {
                "address": 4096,
                "return_type": {"kind": "base", "name": "int"},
                "parameters": [
                    {"name": "fd", "type": {"kind": "base", "name": "int"}},
                    {"name": "buf", "type": {"kind": "pointer", "subtype": {"kind": "base", "name": "void"}}},
                    {"name": "count", "type": {"kind": "base", "name": "int"}}
                ]
            },
            "do_nothing": {
                "address": 8192,
                "return_type": {"kind": "base", "name": "void"},
                "parameters": []
            }
        }
    }

@pytest.fixture
def mock_isf_without_functions():
    """A standard Volatility 3 compliant ISF dictionary missing the functions key."""
    return {
        "metadata": {"format": "6.2.0", "producer": {"name": "test", "version": "1.0"}},
        "base_types": {
            "int": {"size": 4, "signed": True, "kind": "int", "endian": "little"}
        },
        "user_types": {},
        "enums": {},
        "symbols": {}
    }


def test_parse_functions(mock_isf_with_functions):
    """Verify that function signatures are correctly ingested and exposed."""
    dffi = DFFI(mock_isf_with_functions)
    
    assert "sys_read" in dffi.functions
    func = dffi.get_function("sys_read")
    
    assert func is not None
    assert func.name == "sys_read"
    assert func.address == 4096
    assert func.return_type["name"] == "int"
    assert len(func.parameters) == 3


def test_functions_warning_on_missing_key(mock_isf_without_functions, capsys):
    """Verify that accessing .functions on an old ISF prints the specific warning once."""
    dffi = DFFI(mock_isf_without_functions)
    
    # 1. Trigger the warning by accessing the property
    funcs = dffi.functions
    assert len(funcs) == 0
    
    # 2. Capture and verify the exact output format
    captured = capsys.readouterr().out
    assert "Warning: ISF file" in captured
    assert "is missing the 'functions' key. Skipping function loading for this file." in captured
    assert "<dict_" in captured  # Verify it printed the pseudo-path for the dict
    
    # 3. Trigger again to ensure the flag prevents duplicate prints
    print(dffi.functions)
    captured_second = capsys.readouterr().out
    assert captured_second == ""


def test_get_function_silent_fallback(mock_isf_without_functions, capsys):
    """Verify that get_function silently returns None without spamming the warning."""
    dffi = DFFI(mock_isf_without_functions)
    
    # In your logic, get_function does not trigger the warning loop
    func = dffi.get_function("sys_read")
    assert func is None
    
    captured = capsys.readouterr().out
    assert captured == ""


def test_multi_isf_function_resolution(mock_isf_with_functions, mock_isf_without_functions, capsys):
    """Verify that functions can be fetched across multiple loaded ISF profiles, warning appropriately."""
    dffi = DFFI([mock_isf_without_functions, mock_isf_with_functions])
    
    # Requesting the full map should warn about the missing one, but still load from the valid one
    funcs = dffi.functions
    captured = capsys.readouterr().out
    
    assert "Warning: ISF file" in captured
    assert "sys_read" in funcs
    assert "do_nothing" in funcs