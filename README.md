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

## 1️⃣ CFFI-style `cdef`

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

## 2️⃣ Load an ISF (Linux/ELF DWARF)

```python
from dwarffi import DFFI

# Accepts .json or .json.xz
ffi = DFFI("vmlinux_isf.json.xz")

task = ffi.typeof("struct task_struct")
print("task_struct sizeof:", ffi.sizeof(task))

ffi.inspect_layout("struct task_struct")
```

---

## 3️⃣ Load an ISF (Windows PDB-derived / Volatility-style)

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

---

# 🤝 Contributing

Contributions are welcome!

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```
