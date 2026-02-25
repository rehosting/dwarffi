import pytest

from dwarffi.dffi import DFFI


@pytest.fixture
def inspection_env():
    isf = {
        "metadata": {},
        "base_types": {
            "u32": {"kind": "int", "size": 4, "signed": False, "endian": "little"},
            "u8": {"kind": "int", "size": 1, "signed": False, "endian": "little"}
        },
        "user_types": {
            "inner": {
                "kind": "struct", "size": 4,
                "fields": {"val": {"offset": 0, "type": {"kind": "base", "name": "u32"}}}
            },
            "collision_struct": {
                "kind": "struct", "size": 12,
                "fields": {
                    # These field names would collide if they were methods on the instance
                    "pretty_print": {"offset": 0, "type": {"kind": "base", "name": "u32"}},
                    "to_dict": {"offset": 4, "type": {"kind": "base", "name": "u32"}},
                    "nested": {"offset": 8, "type": {"kind": "struct", "name": "inner"}}
                }
            },
            "padded_struct": {
                "kind": "struct", "size": 16,
                "fields": {
                    "a": {"offset": 0, "type": {"kind": "base", "name": "u32"}},
                    # 4-byte gap here
                    "b": {"offset": 8, "type": {"kind": "base", "name": "u32"}}
                    # 4-byte trailing gap
                }
            }
        },
        "enums": {
            "STATUS": {
                "size": 4, "base": "u32",
                "constants": {"OK": 0, "FAIL": 1}
            }
        },
        "symbols": {},
        "typedefs": {}
    }
    return DFFI(isf)

def test_namespace_collision_safety(inspection_env):
    """Verifies we can access fields named exactly like our helper functions."""
    inst = inspection_env.new("struct collision_struct")
    
    # Set values in the 'collision' fields
    inst.pretty_print = 0xAAAA
    inst.to_dict = 0xBBBB
    inst.nested.val = 0xCCCC
    
    # Ensure they are readable and didn't trigger method logic
    assert inst.pretty_print == 0xAAAA
    assert inst.to_dict == 0xBBBB
    
    # Verify the DFFI engine can still inspect it externally
    dump = inspection_env.to_dict(inst)
    assert dump["pretty_print"] == 0xAAAA
    assert dump["to_dict"] == 0xBBBB
    assert dump["nested"]["val"] == 0xCCCC

def test_pretty_print_output(inspection_env, capsys):
    """Checks the formatting of the recursive pretty printer."""
    inst = inspection_env.new("struct collision_struct")
    inst.pretty_print = 1
    inst.to_dict = 2
    inst.nested.val = 3
    
    output = inspection_env.pretty_print(inst)
    
    # Verify structure and values appear in text
    assert "collision_struct {" in output
    assert "pretty_print: 1" in output
    assert "nested: inner {" in output
    assert "val: 3" in output
    assert "}" in output

def test_inspect_layout_padding(inspection_env, capsys):
    """Verifies that padding/gaps are correctly identified in the layout view."""
    inspection_env.inspect_layout("struct padded_struct")
    captured = capsys.readouterr().out
    
    # Expect to see [PADDING] blocks at specific offsets
    assert "0        4      a" in captured
    assert "4        4      [PADDING]" in captured
    assert "8        4      b" in captured
    assert "12       4      [PADDING]" in captured

def test_large_array_preview(inspection_env):
    """Ensures we don't dump 10,000 lines for a giant kernel array."""
    arr = inspection_env.new("u32[1000]")
    output = inspection_env.pretty_print(arr)
    
    # Should show a preview and count, not 1000 items
    assert "... (1000 items)" in output
    assert output.count("0 (u32)") <= 5 # Only a few preview items

def test_enum_to_dict_conversion(inspection_env):
    """Verifies enums are converted to their name or value in dicts."""
    inst = inspection_env.new("enum STATUS")
    inst[0] = "FAIL"
    
    data = inspection_env.to_dict(inst)
    assert data == 1 # The value of FAIL