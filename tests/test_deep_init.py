import pytest
from dwarffi import DFFI

def test_deep_initialization_syscall_hook():
    """
    Verifies that dwarffi can recursively initialize a complex struct containing
    nested structs, string arrays, and arrays of structs using a standard Python dict.
    """
    isf = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "long": {"kind": "int", "size": 8, "signed": True, "endian": "little"},
            "bool": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
            "char": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
        },
        "user_types": {
            "value_filter": {
                "kind": "struct", "size": 40,
                "fields": {
                    "enabled": {"offset": 0, "type": {"kind": "base", "name": "bool"}},
                    "type": {"offset": 4, "type": {"kind": "base", "name": "int"}},
                    "value": {"offset": 8, "type": {"kind": "base", "name": "long"}},
                    "min_value": {"offset": 16, "type": {"kind": "base", "name": "long"}},
                    "max_value": {"offset": 24, "type": {"kind": "base", "name": "long"}},
                    "bitmask": {"offset": 32, "type": {"kind": "base", "name": "long"}},
                }
            },
            "syscall_hook": {
                "kind": "struct", "size": 336,
                "fields": {
                    "enabled": {"offset": 0, "type": {"kind": "base", "name": "bool"}},
                    "on_enter": {"offset": 1, "type": {"kind": "base", "name": "bool"}},
                    "name": {
                        "offset": 2, 
                        "type": {"kind": "array", "count": 16, "subtype": {"kind": "base", "name": "char"}}
                    },
                    "filter_pid": {"offset": 20, "type": {"kind": "base", "name": "int"}},
                    "retval_filter": {"offset": 24, "type": {"kind": "struct", "name": "value_filter"}},
                    "arg_filters": {
                        "offset": 64, 
                        "type": {"kind": "array", "count": 6, "subtype": {"kind": "struct", "name": "value_filter"}}
                    }
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    ffi = DFFI(isf)

    # The structured dict we want to pass
    init_data = {
        "enabled": True,
        "on_enter": False,
        "name": "sys_open",
        "filter_pid": 1337,
        "retval_filter": {
            "enabled": True,
            "type": 0,
            "value": -1, # Expecting an error
            "min_value": 0,
            "max_value": 0,
            "bitmask": 0
        },
        "arg_filters": [
            # Arg 0
            {"enabled": True, "type": 1, "value": 0, "min_value": 10, "max_value": 20, "bitmask": 0},
            # Arg 1
            {"enabled": False, "type": 0, "value": 0, "min_value": 0, "max_value": 0, "bitmask": 0},
            # We omit the rest, expecting them to safely default to 0
        ]
    }

    # Initialize it all in one go
    sch = ffi.new("struct syscall_hook", init_data)

    # 1. Assert basic primitives (bools return 1 or 0 as C-ints)
    assert sch.enabled == 1
    assert sch.on_enter == 0
    assert sch.filter_pid == 1337

    # 2. Assert string assignment
    assert ffi.string(sch.name) == b"sys_open"

    # 3. Assert nested struct
    assert sch.retval_filter.enabled == 1
    assert sch.retval_filter.value == -1

    # 4. Assert array of structs
    assert sch.arg_filters[0].enabled == 1
    assert sch.arg_filters[0].min_value == 10
    assert sch.arg_filters[0].max_value == 20
    
    assert sch.arg_filters[1].enabled == 0
    
    # 5. Assert omitted array elements default to zeroed memory safely
    assert sch.arg_filters[2].enabled == 0
    assert sch.arg_filters[5].value == 0