import pytest

from dwarffi import DFFI, Ptr


@pytest.fixture
def ffi_env(tmp_path):
    import json

    isf_data = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
        },
        "user_types": {},
        "enums": {},
        "symbols": {},
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

def test_ptr_tagged_and_page_alignment(ffi_env: DFFI):
    """
    Simulate AArch64 tagged pointers and OS page alignment masking.
    Verifies that complex bitwise logic chains evaluate correctly.
    """
    d = ffi_env
    # 64-bit pointer with a metadata tag in the top 8 bits
    # Tag: 0xA5, True Address: 0x00007FFF80001234
    tagged_addr = 0xA5007FFF80001234
    ptr = d.t.int.ptr(tagged_addr)

    # 1. Extract the tag (shift right 56 bits)
    tag = (ptr & 0xFF00000000000000) >> 56
    assert tag == 0xA5

    # 2. Clear the tag to get the routable address
    actual_addr = ptr & 0x00FFFFFFFFFFFFFF
    assert actual_addr == 0x00007FFF80001234

    # 3. Calculate 4KB Page alignment (Clear the bottom 12 bits)
    # Python bitwise NOT on standard ints can be tricky with signs, 
    # so we explicitly mask the bits we want.
    PAGE_MASK = 0xFFFFFFFFFFFFF000 
    page_base = actual_addr & PAGE_MASK
    assert page_base == 0x00007FFF80001000

    # 4. Isolate the offset within the page
    page_offset = actual_addr & 0xFFF
    assert page_offset == 0x234
