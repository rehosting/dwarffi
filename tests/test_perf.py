from dwarffi.dffi import DFFI


def test_bulk_unpack_correctness():
    """Verifies that bulk unpack returns the exact same data as field-by-field access."""
    isf = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "short": {"kind": "int", "size": 2, "signed": True, "endian": "little"},
            "byte": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
        },
        "user_types": {
            "fast_struct": {
                "kind": "struct", "size": 8,  # 4 (int) + 2 (short) + 1 (byte) + 1 (padding)
                "fields": {
                    "a": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "b": {"offset": 4, "type": {"kind": "base", "name": "short"}},
                    "c": {"offset": 6, "type": {"kind": "base", "name": "byte"}},
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    ffi = DFFI(isf)
    inst = ffi.new("struct fast_struct", {"a": 1000000, "b": -500, "c": 255})

    # 1. Test Field-by-Field
    manual_data = (inst.a, inst.b, inst.c)
    assert manual_data == (1000000, -500, 255)

    # 2. Test Bulk Unpack
    bulk_data = ffi.unpack(inst)
    assert bulk_data == (1000000, -500, 255)
    
    # 3. Ensure they match exactly
    assert manual_data == bulk_data