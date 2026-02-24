import json
import struct

from dwarffi import DFFI


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

def test_incomplete_type_bridge():
    """Tests resolving a pointer to a struct defined in a different ISF file."""
    # ISF 1: Defines struct A which contains a pointer to struct B (undefined here)
    isf_a = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
        },
        "user_types": {
            "struct_a": {
                "kind": "struct", "size": 8,
                "fields": {
                    "ptr_b": {"offset": 0, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "struct_b"}}}
                }
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {}
    }

    # ISF 2: Defines struct B
    isf_b = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
        },
        "user_types": {
            "struct_b": {
                "kind": "struct", "size": 4,
                "fields": {"val": {"offset": 0, "type": {"kind": "base", "name": "int"}}}
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {}
    }

    ffi = DFFI([isf_a, isf_b])

    # Create struct A and point ptr_b to a buffer containing a struct B
    buf = bytearray(12) # 8 for A, 4 for B
    a = ffi.from_buffer("struct struct_a", buf)
    a.ptr_b = 8 # Offset to where B will be

    # Access B through A's pointer. 
    # This requires DFFI to find 'struct_b' in the second ISF.
    b_ptr = a.ptr_b
    b_resolved = ffi.from_buffer(b_ptr.points_to_type_name, buf, offset=b_ptr.address)
    
    b_resolved.val = 42
    assert b_resolved.val == 42
    assert struct.unpack_from("<i", buf, 8)[0] == 42