# dwarffi

A **debug-symbol-powered type interface for Python**.

`dwarffi` lets you interact with real-world memory layouts using
**Intermediate Symbol Files (ISF)**—portable JSON representations of compiled type information.
It provides a CFFI-like experience without requiring header files: instead, it uses the
structures as they exist in the compiled binary’s debug data.

**ISF files** encode the exact memory layout (offsets, padding, alignment, bitfields, pointer size, endianness)
as defined by the toolchain and target architecture.

- **Linux / embedded workflows:** ISF generated from **DWARF** in ELF binaries (e.g., via `dwarf2json`).
- **Windows workflows:** ISF generated from **PDB** symbols (e.g., Volatility3-style Windows ISFs generated from PDBs).
- **MacOS / Mach-O workflows:** ISF generated from DWARF in Mach-O binaries.


Read more about `dwarf2json` and ISF in the [dwarf2json README](https://github.com/volatilityfoundation/dwarf2json).  

For Windows ISF context, see [Volatility3](https://github.com/volatilityfoundation/volatility3/).

You can also find many symbol tables in ISF format on [Volatility3's symbols repository](https://github.com/volatilityfoundation/volatility3/tree/develop?tab=readme-ov-file#symbol-tables).

This project builds on a tremendous amount of prior work in the volatility community and takes inspiration from projects like `ctypes`, `cffi`, `pyelftools`, and `volatility3`'s symbol handling. The core innovation is the seamless integration of ISF as a first-class type system in Python, with powerful features for live memory access and dynamic type generation.


---

## 🚀 Features

- **ISF-Native**: No header rewriting. Point to a `.json` or `.json.xz` ISF file and use types immediately.
- **Cross-Platform Symbols**:
  - **DWARF-backed ISFs** (common on Linux/ELF, embedded targets)
  - **PDB-backed ISFs** (common for Windows kernel/user-mode analysis toolchains)
- **Architecture & ABI Aware**: Handles big-endian and little-endian layouts transparently, respects pointer width and packing.
- **Dynamic `cdef`**: Compile C code on the fly to generate types, with automatic debug-type retention to prevent compilers from stripping unused definitions.
- **Recursive Typedef Handling**: Automatic resolution and decay of typedef chains.
- **C-Style Magic**:
  - Pointer arithmetic (`ptr + 5`)
  - Pointer subtraction (`ptr2 - ptr1`)
  - Array slicing (`arr[1:5]`)
  - Deep struct initialization via nested dictionaries
- **Safety Semantics**:
  - Automatic bit-masking
  - Sign extension
  - C-style integer overflow/underflow behavior
- **Anonymous Struct/Union Flattening**: Access anonymous union members directly (ideal for register maps).
- **ISF Export Support**: Save dynamically generated ISFs to `.json` or `.json.xz`.
- **Introspection Utilities**:
  - `inspect_layout()` for pahole-style field offsets/padding
  - `pretty_print()` and `to_dict()` for human-readable / JSON-friendly inspection of instances
- **Live Memory Backends**: Interface directly with QEMU, GDB, Volatility, or raw firmware dumps using a simple read/write API.
- **Pointer Chaining**: Recursively dereference pointers (`ptr.deref()`) and stride through remote memory using C-style array indexing (`ptr[5]`).
- **Zero-Copy Performance**: High-performance handler binding that automatically switches between zero-copy native buffer access and backend proxying.
- **Fuzzy Search**: Find symbols and types across massive ISFs using glob or regex patterns.

---

## 📦 Installation

```bash
pip install dwarffi
```

### Requirements for `cdef()`

To use dynamic compilation:

- A C compiler (`gcc`, `clang`, or cross-compiler)
- `dwarf2json` available in your PATH

NOTE: Some compilers may optimize away unused debug types. For example, with `gcc`, use:
`-fno-eliminate-unused-debug-types`.

---

# 🛠️ Quick Start

## Load an ISF (Linux/ELF DWARF)

```python
from dwarffi import DFFI

# Accepts .json or .json.xz
ffi = DFFI("ubuntu:5.4.0-26-generic:64.json.xz")


list_head_type = ffi.typeof("list_head")
print("list_head sizeof:", ffi.sizeof(list_head_type))
print(list_head_type)

''' prints out:
struct list_head (size: 16 bytes) {
  [+0  ] pointer next;
  [+8  ] pointer prev;
}
'''

# make a new complex type
proc = ffi.new("struct task_struct", init={"pid": 1234, "comm": b"my_process"})


print(proc.pid)              # 1234
print(bytes(proc.comm))      # b'my_process\x00\x00\x00\x00\x00\x00'
print(ffi.string(proc.comm)) # b'my_process'
```

Download this example .json.xz [here](https://panda.re/volatility3_profiles/ubuntu:5.4.0-26-generic:64.json.xz).

---

## Load an ISF (Windows PDB-derived / Volatility-style)

```python
from dwarffi import DFFI

# Volatility-style Windows symbols are typically .json.xz ISFs
ffi = DFFI("ntkrnlmp.pdb/<GUID>-<AGE>.json.xz")

le = ffi.typeof("struct _LIST_ENTRY")
buf = bytearray(ffi.sizeof(le))
inst = ffi.from_buffer("struct _LIST_ENTRY", buf)

inst.Flink = 0x1122334455667788
inst.Blink = 0x8877665544332211

print(ffi.pretty_print(inst))
print(ffi.to_dict(inst))

ffi.inspect_layout("struct _UNICODE_STRING")
```

---

## CFFI-style `cdef`

We do support inline C definitions that compile down to DWARF and ISF on the fly. This is ideal for quick prototyping or when you have a small struct definition that isn't already in your ISF.

```python
from dwarffi import DFFI

ffi = DFFI()
ffi.cdef("""
    struct sensor_data {
        uint32_t timestamp;
        int16_t  readings[3];
        uint8_t  status;
    };
""")

sensor = ffi.new("struct sensor_data", {
    "timestamp": 1234567,
    "readings": [10, -5, 20],
    "status": 0x01
})

print(f"Bytes: {ffi.to_bytes(sensor).hex()}")
print(f"Reading[1]: {sensor.readings[1]}")  # -5
```
---


# 🧩 Advanced Usage

## Anonymous Unions

```python
ffi.cdef("""
struct reg_map {
    union {
        uint32_t ALL;
        struct {
            uint16_t LOW;
            uint16_t HIGH;
        };
    };
};
""")

reg = ffi.new("struct reg_map")
reg.ALL = 0x12345678
print(hex(reg.HIGH))  # 0x1234
```

---

## Pointer Arithmetic

```python
ptr = ffi.cast("int *", 0x4000)
next_ptr = ptr + 1
print(hex(next_ptr.address))
```
---

## 🧠 Memory Backends & Live Data
dwarffi can bind to live external memory (debuggers, emulators, or remote targets). Instead of snapshotting memory into a local bytearray, you can use from_address() to interact with the target in real-time.

Using Raw Bytes (Mapping at Address 0)
If you provide raw bytes or a bytearray as a backend, dwarffi treats it as a physical memory map starting at address 0x0.

```python
# firmware.bin is 1MB
with open("firmware.bin", "rb") as f:
    ffi = DFFI(isf, backend=f.read())

# Map a header at its specific physical address
header = ffi.from_address("struct fw_header", 0x4000)
print(f"Magic: {hex(header.magic)}")
```

### Implementing a Custom Backend (e.g., GDB)
You can wrap any memory access API by implementing the MemoryBackend interface.

```python
from dwarffi.backend import MemoryBackend

class GDBBackend(MemoryBackend):
    def read(self, address: int, size: int) -> bytes:
        return gdb.selected_inferior().read_memory(address, size).tobytes()

    def write(self, address: int, data: bytes) -> None:
        gdb.selected_inferior().write_memory(address, data)

ffi = DFFI(isf, backend=GDBBackend())

# Now 'task' reads memory from GDB on-demand when you access fields
task = ffi.from_address("struct task_struct", 0xffff888000000000)
print(f"Current PID: {task.pid}")
```
### Live Pointer Traversal
When a MemoryBackend is configured, Ptr objects become "live." Calling `.deref()` or using array indexing fetches the target memory from the backend automatically.

```python
# Get a pointer to an array of nodes in kernel memory
list_ptr = ffi.from_address("struct node *", 0x2000)

# Chained dereferencing (node->next->next)
# Each .deref() triggers a backend read
third_node = list_ptr.deref().next.deref().next.deref()

# C-style array indexing on the backend
fifth_node = list_ptr[4]
```

### Fuzzy Symbol Discovery
```python
# Find all kernel syscall table entries
syscalls = ffi.search_symbols("__x64_sys_*")

for name, sym in syscalls.items():
    print(f"Found {name} at {hex(sym.address)}")
```
### Walking a Process List

```python
# Simulating a container_of walk through a kernel task list
init_task = ffi.from_address("struct task_struct", ffi.get_symbol("init_task").address)

# Walk the circular 'tasks' list_head
curr_list = init_task.tasks.next.deref()

while curr_list.address != init_task.tasks.address:
    # Use cast with address arithmetic to get the parent task_struct
    task_addr = curr_list.address - ffi.offsetof("struct task_struct", "tasks")
    task = ffi.cast("struct task_struct", task_addr)
    
    print(f"Process: {ffi.string(task.comm)} [PID: {task.pid}]")
    curr_list = curr_list.next.deref()
```

---

# ⚙️ How It Works

`dwarffi` operates in three phases:

### 1. Parsing

Loads one or more ISF files (`.json` or `.json.xz`) that represent a compiled type tree
(e.g., derived from DWARF or PDB symbols).

### 2. Type Synthesis

Builds Python representations of:

- Base types
- Structs / unions
- Enums
- Typedef chains
- Arrays
- Pointers
- Bitfields

### 3. Memory Mapping

Uses Python’s `struct.pack` / `struct.unpack` to:

- Convert Python integers into architecture-accurate byte layouts
- Apply endianness rules and pointer size
- Respect alignment, padding, and bitfield masks

Instances are bound views into `bytearray` buffers. Field access directly
reads/writes into the underlying buffer.

---

# 📚 Core API Reference

### `DFFI(path: str | Path | dict | None = None)`

Create an empty instance or load an ISF (from a dict, `.json`, or `.json.xz`).

### `cdef(source, compiler="gcc", save_isf_to=None)`

Compile C source → DWARF → ISF → load into current FFI.

### `new(ctype, init=None)`

Allocate a new instance of a C type.

### `from_buffer(ctype, buffer)`

Bind a type to existing memory.

### `sizeof(ctype)`

Return size in bytes.

### `offsetof(ctype, field)`

Return byte offset of a field.

### `cast(ctype, value)`

Reinterpret memory or create pointer instances.

### `addressof(instance)`

Return pointer to instance.

### `inspect_layout(ctype)`

Print pahole-style offsets and padding.

### `pretty_print(cdata)`

Recursively format bound instances/arrays/pointers as a readable tree.

### `to_dict(cdata)`

Convert bound instances/arrays/pointers to Python-native structures.

### `from_address(ctype, address)`
Binds a type to a specific address in the configured MemoryBackend. Returns a BoundTypeInstance or a Ptr.

### `search_symbols(pattern, use_regex=False)`

Searches for symbols matching a glob (e.g., *sys_call*) or regex pattern across all loaded ISFs.

### `addressof(instance, *fields)`

Returns a `Ptr` to an instance or a nested field. If the instance is backend-backed, the pointer address will be the absolute address in that backend.

### `Ptr.deref()`

The core dereference operator. If the pointer targets another pointer, it actively resolves the chain by reading from the backend.

---

# 🤝 Contributing

Contributions are welcome!

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
