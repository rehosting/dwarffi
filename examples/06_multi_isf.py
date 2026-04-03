from dwarffi.dffi import DFFI


def test_multi_isf():
    kernel_isf = {
        "metadata": {"format": "1.4.0"},
        "base_types": {"int": {"kind": "int", "size": 4}},
        "user_types": {
            "device_struct": {
                "kind": "struct", "size": 4, 
                "fields": {"id": {"type": {"kind": "base", "name": "int"}, "offset": 0}}
            }
        },
        "symbols": {}, "enums": {}, "typedefs": {} # Add missing required keys
    }

    driver_isf = {
        "metadata": {"format": "1.4.0"},
        "base_types": {}, # Add missing required keys
        "user_types": {
            "my_driver_ctx": {
                "kind": "struct", "size": 8,
                "fields": {
                    "dev": {"type": {"kind": "struct", "name": "device_struct"}, "offset": 0},
                    "state": {"type": {"kind": "base", "name": "int"}, "offset": 4}
                }
            }
        },
        "symbols": {}, "enums": {}, "typedefs": {} # Add missing required keys
    }
    

    # 1. Load both - DFFI resolves 'device_struct' by searching across files
    d = DFFI([kernel_isf, driver_isf])

    # 2. Instantiate type from ISF B that depends on ISF A
    ctx = d.t.my_driver_ctx(state=1, dev={'id': 99})

    # 3. Verify cross-file resolution
    assert ctx.state == 1
    assert ctx.dev.id == 99
    assert d.sizeof(ctx) == 8

    print("Example 06 (Cross-ISF): Success")

if __name__ == "__main__":
    test_multi_isf()