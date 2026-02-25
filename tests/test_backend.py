import pytest

from dwarffi import DFFI
from dwarffi.backend import BytesBackend, LiveMemoryProxy, MemoryBackend
from dwarffi.instances import Ptr


@pytest.fixture
def backend_ffi():
    """Provides a DFFI instance configured with types suitable for testing memory backends."""
    isf = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "char": {"kind": "int", "size": 1, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8}
        },
        "user_types": {
            "node": {
                "kind": "struct",
                "size": 16,
                "fields": {
                    "value": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "next": {"offset": 8, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "node"}}}
                }
            }
        },
        "enums": {},
        "symbols": {}
    }
    return DFFI(isf)

def test_bytes_backend_read_write_boundaries():
    """Test that BytesBackend correctly enforces size bounds and reads/writes bytes."""
    mem = b"HELLO WORLD"
    backend = BytesBackend(mem)
    
    # Valid reads
    assert backend.read(0, 5) == b"HELLO"
    assert backend.read(6, 5) == b"WORLD"
    
    # Out of bounds reads
    with pytest.raises(MemoryError, match="out of bounds"):
        backend.read(6, 10)  # Only 5 bytes left
    with pytest.raises(MemoryError, match="out of bounds"):
        backend.read(100, 1)
    with pytest.raises(MemoryError, match="out of bounds"):
        backend.read(-1, 5)

    # Valid writes
    backend.write(0, b"J")
    assert backend.read(0, 5) == b"JELLO"
    
    # Out of bounds writes
    with pytest.raises(MemoryError, match="out of bounds"):
        backend.write(10, b"!!!") # Writes past end
        
def test_live_memory_proxy():
    """Test that LiveMemoryProxy correctly translates slices into backend read/write calls."""
    backend = BytesBackend(b"\x00" * 20)
    proxy = LiveMemoryProxy(backend)
    
    # Test slice writing
    proxy[5:9] = b"TEST"
    assert backend.read(5, 4) == b"TEST"
    
    # Test slice reading
    assert proxy[5:9] == b"TEST"
    
    # Test single byte reading/writing (integer index)
    proxy[10] = b"X"
    assert proxy[10] == b"X"
    assert backend.read(10, 1) == b"X"

    # Test invalid slice configuration
    with pytest.raises(ValueError, match="requires bounded slices"):
        _ = proxy[5:]

def test_from_address_basic_struct(backend_ffi):
    """Test binding a struct directly to a backend address and modifying live memory."""
    mem = bytearray(0x100)
    backend_ffi.backend = BytesBackend(mem)
    
    # Create struct at 0x50
    node = backend_ffi.from_address("struct node", 0x50)
    
    # Write to live memory via the struct
    node.value = 1337
    
    # Verify the backend memory actually changed
    assert backend_ffi.backend.read(0x50, 4) == (1337).to_bytes(4, "little")
    
    # Read from the backend memory directly
    backend_ffi.backend.write(0x50, (42).to_bytes(4, "little"))
    
    # Verify the struct sees the live change
    assert node.value == 42

def test_from_address_pointer_deref(backend_ffi):
    """Test that from_address returns a Ptr, and that deref() follows it."""
    mem = bytearray(0x3000)
    
    # Setup Node 1 at 0x1000
    mem[0x1000:0x1004] = (100).to_bytes(4, "little")
    mem[0x1008:0x1010] = (0x2000).to_bytes(8, "little") # next points to 0x2000
    
    # Setup Node 2 at 0x2000
    mem[0x2000:0x2004] = (200).to_bytes(4, "little")
    mem[0x2008:0x2010] = (0x0).to_bytes(8, "little")    # next is NULL
    
    backend_ffi.backend = BytesBackend(mem)
    
    # Request a pointer from an address
    node_ptr = backend_ffi.from_address("struct node *", 0x1000)
    assert isinstance(node_ptr, Ptr)
    assert node_ptr.address == 0x1000
    
    # Dereference it
    node1 = node_ptr.deref()
    assert node1.value == 100
    
    # Navigate the pointer chain
    assert isinstance(node1.next, Ptr)
    assert node1.next.address == 0x2000
    
    node2 = node1.next.deref()
    assert node2.value == 200
    assert node2.next.address == 0x0

def test_pointer_array_indexing(backend_ffi):
    """Test that pointers support array-style indexing over a backend."""
    mem = bytearray(0x100)
    
    # Write an array of 3 integers at 0x20
    mem[0x20:0x24] = (11).to_bytes(4, "little")
    mem[0x24:0x28] = (22).to_bytes(4, "little")
    mem[0x28:0x2c] = (33).to_bytes(4, "little")
    
    backend_ffi.backend = BytesBackend(mem)
    
    # Get a pointer to the start of the array
    ptr = backend_ffi.from_address("int *", 0x20)
    
    # Index directly through the pointer
    assert ptr[0] == 11
    assert ptr[1] == 22
    assert ptr[2] == 33
    
    # Test pointer arithmetic
    ptr_shifted = ptr + 2
    assert ptr_shifted.address == 0x28
    assert ptr_shifted.deref() == 33

def test_live_memory_string_reading(backend_ffi):
    """Test that DFFI.string() can read chunked strings from a backend proxy."""
    mem = bytearray(0x100)
    test_str = b"Hello from the memory backend!\x00"
    mem[0x40 : 0x40 + len(test_str)] = test_str
    
    backend_ffi.backend = BytesBackend(mem)
    
    # Bind a char pointer to the string location
    char_ptr = backend_ffi.from_address("char *", 0x40)
    
    # Extract the string
    # We deref the pointer to get the underlying unbounded char array view, 
    # then ask the ffi engine to read the string from it
    result = backend_ffi.string(char_ptr.deref())
    assert result == b"Hello from the memory backend!"
    
    # Test string with maxlen
    result_trunc = backend_ffi.string(char_ptr.deref(), maxlen=5)
    assert result_trunc == b"Hello"

def test_custom_memory_backend(backend_ffi):
    """Test integrating a completely custom mock backend class."""
    
    class CustomDictBackend(MemoryBackend):
        def __init__(self):
            self.memory = {}
            
        def read(self, address: int, size: int) -> bytes:
            result = bytearray(size)
            for i in range(size):
                result[i] = self.memory.get(address + i, 0)
            return bytes(result)
            
        def write(self, address: int, data: bytes) -> None:
            for i, b in enumerate(data):
                self.memory[address + i] = b

    custom = CustomDictBackend()
    backend_ffi.backend = custom
    
    # Write some data via the API
    backend_ffi.from_address("int", 0x100)[0] = 0xdeadbeef
    
    # Verify the custom backend dictionary received the individual byte writes (little endian)
    assert custom.memory[0x100] == 0xef
    assert custom.memory[0x101] == 0xbe
    assert custom.memory[0x102] == 0xad
    assert custom.memory[0x103] == 0xde

def test_unconfigured_backend_raises(backend_ffi):
    """Ensure from_address fails cleanly if no backend is set."""
    backend_ffi.backend = None
    with pytest.raises(RuntimeError, match="No memory backend was configured"):
        backend_ffi.from_address("int", 0x100)

def test_backend_exotic_integer_router_fallback():
    """
    Tests that the smart router successfully falls back to slicing and 
    unpack_from/pack_into for exotic types (like 24-bit integers) that 
    cannot use native struct.pack over a Memory Proxy.
    """
    isf = {
        "metadata": {},
        "base_types": {
            "int24": {"kind": "int", "size": 3, "signed": True, "endian": "little"},
            "int128": {"kind": "int", "size": 16, "signed": False, "endian": "little"}
        },
        "user_types": {
            "exotic_struct": {
                "kind": "struct", "size": 19,
                "fields": {
                    "val24": {"offset": 0, "type": {"kind": "base", "name": "int24"}},
                    "val128": {"offset": 3, "type": {"kind": "base", "name": "int128"}},
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    ffi = DFFI(isf, backend=bytearray(100))
    
    inst = ffi.from_address("struct exotic_struct", 0x10)
    
    # 1. Write exotic sizes to the proxy
    inst.val24 = -8388608  # Min 24-bit signed int
    inst.val128 = 0x112233445566778899AABBCCDDEEFF00
    
    # 2. Read them back through the proxy router
    assert inst.val24 == -8388608
    assert inst.val128 == 0x112233445566778899AABBCCDDEEFF00
    
    # 3. Verify the underlying backend bytes were written exactly as expected
    backend_bytes = ffi.backend.read(0x10, 19)
    assert backend_bytes[0:3] == b"\x00\x00\x80" # -8388608 in 3-byte little endian
    assert backend_bytes[3:19] == b"\x00\xff\xee\xdd\xcc\xbb\xaa\x99\x88\x77\x66\x55\x44\x33\x22\x11"

def test_backend_bitfield_read_write():
    """Tests that bitfields (which do their own internal slicing) work perfectly over a proxy."""
    isf = {
        "metadata": {},
        "base_types": {"uint32": {"kind": "int", "size": 4, "signed": False, "endian": "little"}},
        "user_types": {
            "bf_struct": {
                "kind": "struct", "size": 4,
                "fields": {
                    "flag1": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 3, "bit_position": 0, "type": {"name": "uint32"}}},
                    "flag2": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 10, "bit_position": 3, "type": {"name": "uint32"}}},
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    ffi = DFFI(isf, backend=bytearray(100))
    inst = ffi.from_address("struct bf_struct", 0x20)
    
    inst.flag1 = 5
    inst.flag2 = 500
    
    assert inst.flag1 == 5
    assert inst.flag2 == 500
    
    # (500 << 3) | 5 = 4005 -> 0x0FA5
    assert ffi.backend.read(0x20, 4) == b"\xa5\x0f\x00\x00"

def test_backend_enum_handling():
    """Tests that Enum instances serialize and deserialize seamlessly over the proxy."""
    isf = {
        "metadata": {},
        "base_types": {"int": {"kind": "int", "size": 4, "signed": True, "endian": "little"}},
        "user_types": {},
        "enums": {
            "state_e": {
                "size": 4, "base": "int",
                "constants": {"IDLE": 0, "RUNNING": 1, "DEAD": -1}
            }
        },
        "symbols": {}
    }
    ffi = DFFI(isf, backend=bytearray(100))
    state_ptr = ffi.from_address("enum state_e *", 0x40)
    
    # Dereference enum pointer into live memory
    state = state_ptr.deref()
    
    # Assign via string
    state[0] = "RUNNING"
    assert state[0].name == "RUNNING"
    assert state[0]._value == 1
    assert ffi.backend.read(0x40, 4) == b"\x01\x00\x00\x00"
    
    # Assign via int
    state[0] = -1
    assert state[0].name == "DEAD"
    assert ffi.backend.read(0x40, 4) == b"\xff\xff\xff\xff"

def test_backend_nested_struct_array():
    """Tests walking deeply nested struct arrays backed by live memory."""
    isf = {
        "metadata": {},
        "base_types": {"int": {"kind": "int", "size": 4, "signed": True, "endian": "little"}},
        "user_types": {
            "point": {
                "kind": "struct", "size": 8,
                "fields": {
                    "x": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "y": {"offset": 4, "type": {"kind": "base", "name": "int"}},
                }
            },
            "polygon": {
                "kind": "struct", "size": 24, # 3 points
                "fields": {
                    "points": {"offset": 0, "type": {"kind": "array", "count": 3, "subtype": {"kind": "struct", "name": "point"}}}
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    ffi = DFFI(isf, backend=bytearray(100))
    poly = ffi.from_address("struct polygon", 0x10)
    
    poly.points[0].x = 10
    poly.points[0].y = 20
    poly.points[2].x = 90
    poly.points[2].y = 100
    
    assert poly.points[0].x == 10
    assert poly.points[2].y == 100
    
    # Verify memory
    # point 0: 10, 20
    assert ffi.backend.read(0x10, 8) == (10).to_bytes(4, "little") + (20).to_bytes(4, "little")
    # point 2: 90, 100
    assert ffi.backend.read(0x20, 8) == (90).to_bytes(4, "little") + (100).to_bytes(4, "little")

def test_backend_string_no_null_terminator():
    """Tests that string chunked reading correctly halts at the backend limits if no null is found."""
    # Backend size is exactly 5 bytes. No null terminator present.
    mem = bytearray(b"HELLO")
    ffi = DFFI({"metadata": {}, "base_types": {"char": {"kind": "int", "size": 1, "signed": True, "endian": "little"}}, "user_types": {}, "enums": {}, "symbols": {}})
    ffi.backend = BytesBackend(mem)
    
    char_ptr = ffi.from_address("char *", 0x0)
    
    # The chunk reader should hit the MemoryError boundary and gracefully return what it has, 
    # or raise an exception depending on implementation. In our DFFI implementation, 
    # it catches the end of the buffer by checking chunk sizes.
    # Because BytesBackend strictly raises MemoryError, we expect this to raise when chunking hits the wall.
    with pytest.raises(MemoryError):
        _ = ffi.string(char_ptr.deref())
        
    # But if we limit the maxlen, it should succeed without hitting the boundary!
    assert ffi.string(char_ptr.deref(), maxlen=5) == b"HELLO"

def test_backend_read_only_simulation():
    """Mocks a read-only hardware memory backend to ensure writes are caught."""
    class ReadOnlyBackend(MemoryBackend):
        def __init__(self, data):
            self.data = data
        def read(self, address, size):
            return self.data[address : address+size]
        def write(self, address, data):
            raise PermissionError("Hardware segmentation fault: Read Only Memory")
            
    isf = {"metadata": {}, "base_types": {"int": {"kind": "int", "size": 4, "signed": True, "endian": "little"}}, "user_types": {}, "enums": {}, "symbols": {}}
    ffi = DFFI(isf)
    ffi.backend = ReadOnlyBackend(b"\x00" * 100)
    
    inst = ffi.from_address("int", 0x10)
    
    # Read should succeed
    assert inst[0] == 0
    
    # Write should fail natively
    with pytest.raises(PermissionError, match="Hardware segmentation fault"):
        inst[0] = 5

def test_live_memory_proxy_edge_cases():
    """Tests edge cases specifically on the proxy's __getitem__/__setitem__."""
    backend = BytesBackend(b"\x00" * 10)
    proxy = LiveMemoryProxy(backend)
    
    # Test assigning a single byte via integer index
    proxy[3] = b"\xFF"
    assert proxy[3] == b"\xFF"
    assert backend.read(3, 1) == b"\xFF"
    
    # Test len fakeout
    assert len(proxy) > 1000000000 
    
    # Test invalid index type
    with pytest.raises(TypeError):
        _ = proxy["invalid"]
        
    with pytest.raises(TypeError):
        proxy["invalid"] = b"\x00"

def test_backend_addressof_routing(backend_ffi):
    """
    Tests that addressof correctly calculates physical/virtual addresses
    in backend memory and returns valid dereferenceable pointers.
    """
    mem = bytearray(0x1000)
    
    # Write Node 2 at 0x500
    mem[0x500:0x504] = (999).to_bytes(4, "little")
    
    # Write Node 1 at 0x100, pointing 'next' to 0x500
    mem[0x100:0x104] = (42).to_bytes(4, "little")
    mem[0x108:0x110] = (0x500).to_bytes(8, "little")
    
    backend_ffi.backend = BytesBackend(mem)
    
    # Map Node 1
    node1 = backend_ffi.from_address("struct node", 0x100)
    
    # 1. Base address of the struct
    ptr_to_node1 = backend_ffi.addressof(node1)
    assert ptr_to_node1.address == 0x100
    
    # 2. Address of a nested field (the 'next' pointer is at offset 8)
    ptr_to_next_field = backend_ffi.addressof(node1, "next")
    assert ptr_to_next_field.address == 0x108
    
    # 3. We can dereference that nested pointer pointer!
    # ptr_to_next_field is a `struct node **`
    # deref() gets the actual `struct node *`
    # deref() again gets the actual `struct node` at 0x500!
    node2 = ptr_to_next_field.deref().deref()
    
    assert node2.value == 999