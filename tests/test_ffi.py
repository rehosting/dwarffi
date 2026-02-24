import json
import struct

import pytest

from dwarffi.dffi import DFFI


@pytest.fixture
def ffi_env(tmp_path):
    """Creates a temporary ISF JSON file and loads it into a DFFI object."""
    isf_data = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
            "char": {"kind": "char", "size": 1, "signed": False, "endian": "little"},
            "unsigned long": {"kind": "int", "size": 8, "signed": False, "endian": "little"},
        },
        "user_types": {
            "task_struct": {
                "kind": "struct",
                "size": 16,
                "fields": {
                    "pid": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "state": {"offset": 4, "type": {"kind": "enum", "name": "task_state"}},
                    "flags": {"offset": 8, "type": {"kind": "base", "name": "int"}},
                },
            }
        },
        "enums": {
            "task_state": {
                "size": 4,
                "base": "int",
                "constants": {"TASK_RUNNING": 0, "TASK_INTERRUPTIBLE": 1, "TASK_DEAD": 2},
            }
        },
        "symbols": {},
    }

    isf_file = tmp_path / "test.isf.json"
    with open(isf_file, "w") as f:
        json.dump(isf_data, f)

    return DFFI(str(isf_file))


def test_sizeof_and_offsetof(ffi_env: DFFI):
    assert ffi_env.sizeof("int") == 4
    assert ffi_env.sizeof("task_struct") == 16
    assert ffi_env.sizeof("struct task_struct") == 16
    assert ffi_env.offsetof("struct task_struct", "pid") == 0
    assert ffi_env.offsetof("struct task_struct", "state") == 4


def test_ffi_new_with_initialization(ffi_env: DFFI):
    # Test struct initialization
    task = ffi_env.new("struct task_struct", {"pid": 1337, "state": "TASK_INTERRUPTIBLE"})

    assert task.pid == 1337
    assert task.state.name == "TASK_INTERRUPTIBLE"
    assert int(task.state) == 1
    assert task.flags == 0  # Uninitialized fields should be zeroed

    # Test base type initialization
    my_int = ffi_env.new("int", 42)
    assert int(my_int) == 42
    assert my_int[0] == 42


def test_ffi_cast(ffi_env: DFFI):
    # Cast int to primitive
    primitive = ffi_env.cast("int", -5)
    assert primitive[0] == -5
    assert int(primitive) == -5

    # Cast int to pointer
    ptr = ffi_env.cast("struct task_struct *", 0xC0000000)
    assert ptr.address == 0xC0000000
    # FIX: Expect the full string that was parsed before the '*'
    assert ptr.points_to_type_name == "task_struct"


def test_ffi_from_buffer(ffi_env: DFFI):
    # Create raw bytes and bind them
    raw_memory = bytearray(16)
    struct.pack_into("<i", raw_memory, 0, 9999)  # pid
    struct.pack_into("<i", raw_memory, 4, 2)  # state (TASK_DEAD)

    task = ffi_env.from_buffer("struct task_struct", raw_memory)
    assert task.pid == 9999
    assert task.state.name == "TASK_DEAD"

    # Modifying the wrapper modifies the original buffer
    task.flags = 255
    assert struct.unpack_from("<i", raw_memory, 8)[0] == 255


def test_ffi_buffer_and_memmove(ffi_env: DFFI):
    task1 = ffi_env.new("struct task_struct", {"pid": 100, "flags": 5})
    task2 = ffi_env.new("struct task_struct")

    # memmove task1 into task2
    ffi_env.memmove(task2, task1, ffi_env.sizeof("struct task_struct"))

    assert task2.pid == 100
    assert task2.flags == 5

    # buffer() should expose the memoryview
    view = ffi_env.buffer(task2)
    assert len(view) == 16
    assert view[0:4] == struct.pack("<i", 100)

def test_bytes_casting_and_equality(ffi_env):
    # Assuming ffi_env provides a basic 'int' of 4 bytes
    inst = ffi_env.new("int", 0x11223344)
    
    # 1. Test native bytes() cast
    raw_bytes = bytes(inst)
    assert raw_bytes == b"\x44\x33\x22\x11"  # Assuming Little Endian
    
    # 2. Test direct equality against bytes/bytearray (Zero-copy check)
    assert inst == b"\x44\x33\x22\x11"
    assert inst == bytearray(b"\x44\x33\x22\x11")
    
    # 3. Ensure inequality works
    assert inst != b"\x00\x00\x00\x00"
    
    # 4. Size mismatches evaluate to False, not crash
    assert inst != b"\x44\x33\x22"


def test_ffi_string_maxlen_and_unterminated(ffi_env):
    # 1. Test unterminated string (should read up to end of buffer)
    # create a char array of size 4 without a null terminator
    unterminated = ffi_env.new("char[4]", b"abcd")
    assert ffi_env.string(unterminated) == b"abcd"

    # 2. Test maxlen truncation
    long_str = ffi_env.new("char[20]", b"hello world")
    
    # Normally reads up to the null byte
    assert ffi_env.string(long_str) == b"hello world"
    
    # Maxlen cuts it short
    assert ffi_env.string(long_str, maxlen=5) == b"hello"

def test_typeof_string_parsing(ffi_env):
    # Dynamic Array
    t_array = ffi_env.typeof("int[15]")
    assert t_array["kind"] == "array"
    assert t_array["count"] == 15
    assert t_array["subtype"]["name"] == "int"

    # Pointer parsing
    t_ptr = ffi_env.typeof("struct task_struct *")
    assert t_ptr["kind"] == "pointer"
    assert t_ptr["subtype"]["name"] == "task_struct"

    # Pointer to array isn't explicitly natively supported by the regex yet, 
    # but base cases should parse safely.
    t_base = ffi_env.typeof("  unsigned long  ") # strip spaces
    assert t_base.name == "unsigned long"

def test_ffi_memmove_structs(ffi_env):
    # Allocate two task_structs
    src = ffi_env.new("struct task_struct", {"pid": 5050, "flags": 0xABCD})
    dst = ffi_env.new("struct task_struct")
    
    assert dst.pid == 0
    assert dst.flags == 0
    
    # Memmove exact size
    ffi_env.memmove(dst, src, ffi_env.sizeof("struct task_struct"))
    
    # Verify exact byte copy
    assert dst.pid == 5050
    assert dst.flags == 0xABCD
    assert bytes(src) == bytes(dst)
    
def test_ffi_memmove_raw_bytes(ffi_env):
    dst = ffi_env.new("struct task_struct")
    raw_data = b"\x01\x00\x00\x00\x02\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00"
    
    # Memmove raw bytes into struct buffer
    ffi_env.memmove(dst, raw_data, 12)
    
    assert dst.pid == 1
    assert dst.state.name == "TASK_DEAD" # State 2
    assert dst.flags == 3

def test_linked_list_traversal(ffi_env):
    """Simulate traversing a linked list by casting pointer addresses to struct instances."""
    node_def = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
        },
        "user_types": {
            "node": {
                "kind": "struct", "size": 16,
                "fields": {
                    "val": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "next": {"offset": 8, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "node"}}}
                }
            }
        },
        "enums": {}, "symbols": {}, "typedefs": {}
    }
    # Using the new direct dictionary loading
    ffi_env.load_isf(node_def)

    buf = bytearray(32)
    node_a = ffi_env.from_buffer("struct node", buf, offset=0)
    node_a.val = 101
    node_a.next = 16 # Point to offset 16

    # Bind node_b to the SAME buffer with an offset (Zero-copy)
    node_b = ffi_env.from_buffer("struct node", buf, offset=16)
    node_b.val = 202
    node_b.next = 0

    assert node_a.val == 101
    ptr_to_b = node_a.next
    
    # Resolve node_b using the pointer's address as an offset
    node_b_final = ffi_env.from_buffer("struct node", buf, offset=ptr_to_b.address)
    assert node_b_final.val == 202 # Success!