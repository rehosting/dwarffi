import json

from dwarffi.dffi import DFFI


def test_cross_file_type_resolution(tmp_path):
    # File 1: Defines base types and a core struct
    core_isf = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
        },
        "user_types": {
            "core_struct": {
                "kind": "struct",
                "size": 4,
                "fields": {
                    "magic": {"offset": 0, "type": {"kind": "base", "name": "int"}}
                }
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {}
    }
    
    # File 2: Defines a plugin struct that relies on core_struct from File 1
    plugin_isf = {
        "metadata": {},
        "base_types": {},
        "user_types": {
            "plugin_struct": {
                "kind": "struct",
                "size": 8,
                "fields": {
                    "id": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    # Reference to a type NOT defined in this file
                    "core": {"offset": 4, "type": {"kind": "struct", "name": "core_struct"}}
                }
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {}
    }

    core_file = tmp_path / "core.json"
    plugin_file = tmp_path / "plugin.json"
    
    with open(core_file, "w") as f: 
        json.dump(core_isf, f)
    with open(plugin_file, "w") as f: 
        json.dump(plugin_isf, f)

    ffi = DFFI()
    # Load them sequentially
    ffi.load_isf(str(core_file))
    ffi.load_isf(str(plugin_file))

    # Sizeof should successfully traverse files to resolve 'core_struct'
    assert ffi.sizeof("struct plugin_struct") == 8
    
    # Instantiation should also bridge the files
    inst = ffi.new("struct plugin_struct", {"id": 99, "core": {"magic": 0x12345678}})
    assert inst.id == 99
    assert inst.core.magic == 0x12345678