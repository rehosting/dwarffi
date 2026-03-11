from dwarffi import DFFI


class RecordingBackend:
    def __init__(self, mem: bytearray):
        self.mem = mem
        self.reads = []
        self.writes = []

    def read(self, address: int, size: int) -> bytes:
        self.reads.append((address, size))
        return bytes(self.mem[address:address+size])

    def write(self, address: int, data: bytes) -> None:
        self.writes.append((address, bytes(data)))
        self.mem[address:address+len(data)] = data

def _isf_ptr_struct():
    return {
        "metadata": {},
        "base_types": {
            "u32": {"kind": "int", "size": 4, "signed": False, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
            "void": {"kind": "void", "size": 0, "signed": False, "endian": "little"},
        },
        "user_types": {
            "node": {
                "kind": "struct",
                "size": 16,
                "fields": {
                    "val": {"offset": 0, "type": {"kind": "base", "name": "u32"}},
                    "next": {"offset": 8, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "node"}}},
                },
            }
        },
        "enums": {},
        "symbols": {},
    }

def test_backend_read_write_sizes_and_addresses():
    ffi = DFFI(_isf_ptr_struct())
    mem = bytearray(b"\x00" * 0x100)
    backend = RecordingBackend(mem)
    ffi.backend = backend

    # bind a node at address 0x20
    n = ffi.from_address("struct node", 0x20)

    # write should call backend.write with 4 bytes at 0x20
    n.val = 0xDEADBEEF
    assert backend.writes[-1][0] == 0x20
    assert len(backend.writes[-1][1]) == 4

    # read should call backend.read with 4 bytes at 0x20
    _ = n.val
    assert backend.reads[-1] == (0x20, 4)
