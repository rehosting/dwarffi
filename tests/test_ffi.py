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
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"}
        },
        "user_types": {
            "task_struct": {
                "kind": "struct",
                "size": 16,
                "fields": {
                    "pid": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "state": {"offset": 4, "type": {"kind": "enum", "name": "task_state"}},
                    "flags": {"offset": 8, "type": {"kind": "base", "name": "int"}}
                }
            }
        },
        "enums": {
            "task_state": {
                "size": 4, "base": "int",
                "constants": {"TASK_RUNNING": 0, "TASK_INTERRUPTIBLE": 1, "TASK_DEAD": 2}
            }
        },
        "symbols": {}
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
    ptr = ffi_env.cast("struct task_struct *", 0xc0000000)
    assert ptr.address == 0xc0000000
    # FIX: Expect the full string that was parsed before the '*'
    assert ptr.points_to_type_name == "task_struct"

def test_ffi_from_buffer(ffi_env: DFFI):
    # Create raw bytes and bind them
    raw_memory = bytearray(16)
    struct.pack_into("<i", raw_memory, 0, 9999)  # pid
    struct.pack_into("<i", raw_memory, 4, 2)     # state (TASK_DEAD)
    
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