import pytest
import re
from dwarffi import DFFI, Ptr

@pytest.fixture
def rich_ffi_env():
    """Provides a complex ISF environment specifically for testing introspection and edge cases."""
    isf = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "char": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
        },
        "user_types": {
            "list_head": {
                "kind": "struct", "size": 16,
                "fields": {
                    "next": {"offset": 0, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "list_head"}}},
                    "prev": {"offset": 8, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "list_head"}}}
                }
            },
            "process_node": {
                "kind": "struct", "size": 24,
                "fields": {
                    "pid": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "links": {"offset": 8, "type": {"kind": "struct", "name": "list_head"}}
                }
            }
        },
        "enums": {
            "status_e": {
                "size": 4, "base": "int",
                "constants": {"OK": 0, "ERROR": -1}
            }
        },
        "symbols": {
            "init_task": {"address": 0xffffffff81000000, "type_info": {"kind": "struct", "name": "process_node"}},
            "sys_open": {"address": 0xffffffff81000020, "type_info": {"kind": "function"}},
            "sys_read": {"address": 0xffffffff81000040, "type_info": {"kind": "function"}}
        }
    }
    return DFFI(isf)

# --- 1. ERROR HANDLING & TYPE VALIDATION ---

def test_typeof_invalid_inputs(rich_ffi_env):
    """Hits the TypeError paths in typeof()."""
    with pytest.raises(TypeError, match="Expected string, BoundTypeInstance, Ptr, or BoundArrayView"):
        rich_ffi_env.typeof(12345)
        
    # typeof() safely returns None for missing types
    assert rich_ffi_env.typeof("does_not_exist") is None
    
    # consuming APIs like sizeof() use _typeof_or_raise and raise KeyError
    with pytest.raises(KeyError, match="Unknown type 'does_not_exist'"):
        rich_ffi_env.sizeof("does_not_exist")
        
    # new() explicitly catches the lack of size and raises a ValueError first
    with pytest.raises(ValueError, match="Cannot allocate memory for type 'does_not_exist' with unknown size"):
        rich_ffi_env.new("does_not_exist")

def test_sizeof_errors(rich_ffi_env):
    """Hits the size calculation error paths."""
    with pytest.raises(TypeError, match="Cannot determine size"):
        rich_ffi_env.sizeof(12345)

def test_offsetof_errors(rich_ffi_env):
    """Hits the struct field offset error paths."""
    # Not a struct
    with pytest.raises(TypeError, match="is not a struct or union"):
        rich_ffi_env.offsetof("int", "foo")
        
    # Field doesn't exist
    with pytest.raises(KeyError, match="has no field 'missing'"):
        rich_ffi_env.offsetof("struct process_node", "missing")
        
    # Digging into a non-struct field
    with pytest.raises(TypeError, match="Cannot get offset of 'bad' inside non-struct type"):
        rich_ffi_env.offsetof("struct process_node", "pid", "bad")

def test_cast_errors(rich_ffi_env):
    """Hits invalid casting paths."""
    with pytest.raises(TypeError, match="Cannot cast <class 'list'>"):
        rich_ffi_env.cast("int", [1, 2, 3])

# --- 2. INTROSPECTION, SEARCHING & LAYOUT ---

def test_search_symbols_and_types(rich_ffi_env):
    """Exercises the glob and regex search utilities."""
    # Glob search
    sys_funcs = rich_ffi_env.search_symbols("sys_*")
    assert "sys_open" in sys_funcs
    assert "sys_read" in sys_funcs
    assert "init_task" not in sys_funcs
    
    # Regex search
    regex_funcs = rich_ffi_env.search_symbols(r"^sys_(open|read)$", use_regex=True)
    assert len(regex_funcs) == 2

    # Type glob search
    node_types = rich_ffi_env.search_types("*node")
    assert "process_node" in node_types

def test_find_types_with_member(rich_ffi_env):
    """Tests reverse-lookup of struct members (great for finding container_of targets)."""
    containers = rich_ffi_env.find_types_with_member("links")
    assert "process_node" in containers
    assert "list_head" not in containers

def test_inspect_layout_output(rich_ffi_env, capsys):
    """Ensures inspect_layout doesn't crash and formats correctly."""
    rich_ffi_env.inspect_layout("struct process_node")
    captured = capsys.readouterr().out
    
    assert "Layout of struct process_node" in captured
    assert "pid" in captured
    assert "links" in captured
    assert "[PADDING]" in captured  # pid (4 bytes) + 4 bytes padding to reach links (offset 8)
    
    # Test primitive fallback
    rich_ffi_env.inspect_layout("int")
    captured_prim = capsys.readouterr().out
    assert "(Primitive)" in captured_prim

# --- 3. DICTIONARY EXPORT & PRETTY PRINTING ---

def test_to_dict_conversion(rich_ffi_env):
    """Tests recursive struct/array extraction to pure Python dictionaries."""
    inst = rich_ffi_env.new("struct process_node")
    inst.pid = 1337
    inst.links.next = 0xDEADBEEF
    
    py_dict = rich_ffi_env.to_dict(inst)
    
    assert isinstance(py_dict, dict)
    assert py_dict["pid"] == 1337
    assert py_dict["links"]["next"] == 0xDEADBEEF  # Pointer decayed to integer address
    
    # Test Array export
    arr = rich_ffi_env.new("int[3]", [10, 20, 30])
    py_list = rich_ffi_env.to_dict(arr)
    assert py_list == [10, 20, 30]

def test_pretty_print_formatting(rich_ffi_env):
    """Tests the string formatting generation."""
    inst = rich_ffi_env.new("struct process_node")
    inst.pid = 99
    
    output = rich_ffi_env.pretty_print(inst)
    assert "process_node {" in output
    assert "pid: 99" in output

    # Test array threshold formatting
    large_arr = rich_ffi_env.new("int[20]")
    large_output = rich_ffi_env.pretty_print(large_arr)
    assert "(20 items)" in large_output

# --- 4. ADVANCED INITIALIZATION (_deep_init) ---

def test_deep_init_recursive_population(rich_ffi_env):
    """Tests creating complex structs entirely from nested Python dictionaries."""
    # We populate the outer struct and the inner struct in one call
    init_data = {
        "pid": 5555,
        "links": {
            "next": 0x1000,
            "prev": 0x2000
        }
    }
    
    inst = rich_ffi_env.new("struct process_node", init_data)
    
    assert inst.pid == 5555
    assert inst.links.next.address == 0x1000
    assert inst.links.prev.address == 0x2000

# --- 5. SYMBOL SHIFTING (ASLR/Module Base Addresses) ---

def test_shift_symbol_addresses(rich_ffi_env):
    """Tests that ASLR offsets are correctly applied to the symbol table."""
    # Before shift
    assert rich_ffi_env.symbols["init_task"].address == 0xffffffff81000000
    
    # Apply a 0x1000 slide
    rich_ffi_env.shift_symbol_addresses(0x1000)
    
    # After shift
    assert rich_ffi_env.symbols["init_task"].address == 0xffffffff81001000
    assert rich_ffi_env.symbols["sys_open"].address == 0xffffffff81001020

    # Ensure get_function_address reflects it
    assert rich_ffi_env.get_function_address("init_task") == 0xffffffff81001000

# --- 6. UNPACKING COMPLEX STRUCTS ---

def test_unpack_complex_struct_fails(rich_ffi_env):
    """Structs with pointers/nested structs cannot be fast-path unpacked via struct module."""
    inst = rich_ffi_env.new("struct process_node")
    
    with pytest.raises(TypeError, match="contains complex types, unions, or overlapping fields"):
        rich_ffi_env.unpack(inst)

def test_cdef_dwarf2json_missing(monkeypatch):
    """Tests the cdef error path when dwarf2json isn't installed."""
    # Mock shutil.which to pretend dwarf2json doesn't exist
    import shutil
    original_which = shutil.which
    
    def mock_which(cmd, *args, **kwargs):
        if cmd == "dwarf2json":
            return None
        return original_which(cmd, *args, **kwargs)
        
    monkeypatch.setattr(shutil, "which", mock_which)
    
    ffi = DFFI()
    with pytest.raises(RuntimeError, match="'dwarf2json' not found in PATH"):
        ffi.cdef("struct x { int y; };")