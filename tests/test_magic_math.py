import pytest
from dwarffi.dffi import DFFI

@pytest.fixture
def ffi_env(tmp_path):
    import json
    isf_data = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"}
        },
        "user_types": {},
        "enums": {}, "symbols": {}
    }
    isf_file = tmp_path / "test_magic.json"
    with open(isf_file, "w") as f:
        json.dump(isf_data, f)
    return DFFI(str(isf_file))

def test_pointer_arithmetic(ffi_env: DFFI):
    # Base pointer at 0x1000
    p1 = ffi_env.cast("int *", 0x1000)
    
    # 1. Addition (advances by sizeof(int) = 4)
    p2 = p1 + 5
    assert p2.address == 0x1000 + (5 * 4)
    
    # 2. Subtraction (rewinds)
    p3 = p2 - 2
    assert p3.address == 0x1000 + (3 * 4)
    
    # 3. Pointer-to-Pointer subtraction (returns distance in elements)
    distance = p2 - p1
    assert distance == 5
    
    # 4. Comparisons
    assert p2 > p1
    assert p1 < p3
    assert p2 >= 0x1000

def test_array_decay_and_equality(ffi_env: DFFI):
    arr = ffi_env.new("int[10]", [0, 10, 20, 30, 40])
    
    # 1. Array Equality
    assert arr[:5] == [0, 10, 20, 30, 40]
    
    # 2. Decay to pointer
    ptr = arr + 2
    assert isinstance(ptr, type(ffi_env.cast("int *", 0)))
    
    # Since `arr` is at offset 0, and we added 2 ints (2 * 4 bytes)
    assert ptr.address == 8
    assert ptr.points_to_type_name == "int"

def test_primitive_math(ffi_env: DFFI):
    val = ffi_env.new("int", 100)
    
    # Primitives act like integers naturally
    assert val + 50 == 150
    assert 200 - val == 100
    assert val * 2 == 200
    
    # Comparisons
    assert val > 50
    assert val <= 100
    assert val == 100