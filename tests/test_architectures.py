import json
import struct

import pytest

from dwarffi import DFFI


def make_arch_isf(tmp_path, filename, base_types):
    """Helper to generate an ISF file mimicking a specific architecture."""
    ptr_size = base_types["pointer"]["size"]
    # We create a generic struct to test memory layouts across architectures
    isf_data = {
        "metadata": {"producer": {"name": "test_arch"}},
        "base_types": base_types,
        "user_types": {
            "cpu_context": {
                "kind": "struct",
                # Size dynamically calculated based on pointer size + int size (4)
                "size": ptr_size + 4,
                "fields": {
                    "pc": {  # Program Counter (Pointer)
                        "offset": 0,
                        "type": {"kind": "pointer", "subtype": {"kind": "base", "name": "int"}},
                    },
                    "status": {  # Status Register (Int)
                        "offset": ptr_size,
                        "type": {"kind": "base", "name": "int"},
                    },
                },
            }
        },
        "enums": {},
        "symbols": {},
    }

    filepath = tmp_path / filename
    with open(filepath, "w") as f:
        json.dump(isf_data, f)
    return str(filepath)


# ==============================================================================
# Architecture Fixtures
# ==============================================================================


@pytest.fixture
def ffi_arm32_le(tmp_path):
    """32-bit Little Endian (e.g., ARM Cortex-M, x86)"""
    path = make_arch_isf(
        tmp_path,
        "arm32_le.json",
        {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 4, "endian": "little"},
        },
    )
    return DFFI(path)


@pytest.fixture
def ffi_mips32_be(tmp_path):
    """32-bit Big Endian (e.g., MIPS32, PowerPC 32)"""
    path = make_arch_isf(
        tmp_path,
        "mips32_be.json",
        {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "big"},
            "pointer": {"kind": "pointer", "size": 4, "endian": "big"},
        },
    )
    return DFFI(path)


@pytest.fixture
def ffi_ppc64_be(tmp_path):
    """64-bit Big Endian (e.g., PowerPC64, SPARC64)"""
    path = make_arch_isf(
        tmp_path,
        "ppc64_be.json",
        {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "big"},
            "long": {"kind": "int", "size": 8, "signed": True, "endian": "big"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "big"},
        },
    )
    return DFFI(path)


# ==============================================================================
# Tests
# ==============================================================================


def test_32bit_little_endian_memory_layout(ffi_arm32_le: DFFI):
    assert ffi_arm32_le.sizeof("pointer") == 4
    assert ffi_arm32_le.sizeof("struct cpu_context") == 8

    # 1. Allocate natively via DFFI
    ctx = ffi_arm32_le.new("struct cpu_context")
    ctx.pc = 0x08001234
    ctx.status = 0xDEADBEEF

    # 2. Check the raw bytearray is correctly Little Endian packed
    raw_bytes = ffi_arm32_le.to_bytes(ctx)
    assert len(raw_bytes) == 8

    # '<I' and '<i' are Little-Endian 32-bit unsigned/signed respectively
    assert struct.unpack("<I", raw_bytes[0:4])[0] == 0x08001234

    # Note: 0xDEADBEEF is negative as a signed 32-bit integer (-559038737)
    assert (
        struct.unpack("<i", raw_bytes[4:8])[0]
        == struct.unpack("<i", struct.pack("<I", 0xDEADBEEF))[0]
    )


def test_32bit_big_endian_memory_layout(ffi_mips32_be: DFFI):
    assert ffi_mips32_be.sizeof("pointer") == 4
    assert ffi_mips32_be.sizeof("struct cpu_context") == 8

    # 1. Provide a raw memory buffer from an emulated Big Endian environment
    # >I = Big Endian Unsigned Int, >i = Big Endian Signed Int
    raw_memory = bytearray()
    raw_memory += struct.pack(">I", 0x8000AABB)  # pc
    raw_memory += struct.pack(">i", 1337)  # status

    # 2. Bind the buffer using DFFI
    ctx = ffi_mips32_be.from_buffer("struct cpu_context", raw_memory)

    # 3. Verify DFFI correctly interprets the Big Endian bytes
    assert ctx.pc.address == 0x8000AABB
    assert ctx.status == 1337

    # 4. Modify via DFFI and ensure the underlying buffer updates in Big Endian
    ctx.status = -5
    assert struct.unpack(">i", raw_memory[4:8])[0] == -5


def test_64bit_big_endian_memory_layout(ffi_ppc64_be: DFFI):
    assert ffi_ppc64_be.sizeof("pointer") == 8
    assert ffi_ppc64_be.sizeof("struct cpu_context") == 12  # 8 byte pointer + 4 byte int

    # Set up raw memory for 64-bit BE
    # >Q = Big Endian Unsigned Long Long (64-bit)
    # >i = Big Endian Signed Int (32-bit)
    raw_memory = bytearray()
    raw_memory += struct.pack(">Q", 0xFFFFFFFF00001111)  # pc (64-bit)
    raw_memory += struct.pack(">i", 4096)  # status (32-bit)

    ctx = ffi_ppc64_be.from_buffer("struct cpu_context", raw_memory)

    # Verify parsing
    assert ctx.pc.address == 0xFFFFFFFF00001111
    assert ctx.status == 4096

    # Update properties
    ctx.pc = 0x1111222233334444
    ctx.status = -999

    # Verify underlying bytearray packing
    assert struct.unpack(">Q", raw_memory[0:8])[0] == 0x1111222233334444
    assert struct.unpack(">i", raw_memory[8:12])[0] == -999


def test_base_type_casting_endianness(ffi_ppc64_be: DFFI, ffi_arm32_le: DFFI):
    # Test casting an integer to a Big Endian long
    be_long = ffi_ppc64_be.cast("long", 0x1122334455667788)
    assert be_long[0] == 0x1122334455667788
    # Big Endian means most significant byte comes first
    assert ffi_ppc64_be.to_bytes(be_long)[0] == 0x11

    # Test casting an integer to a Little Endian int
    le_int = ffi_arm32_le.cast("int", 0x11223344)
    assert le_int[0] == 0x11223344
    # Little Endian means least significant byte comes first
    assert ffi_arm32_le.to_bytes(le_int)[0] == 0x44


def test_primitive_bounds_and_signedness(ffi_arm32_le: DFFI):
    # Ensure signed values roundtrip correctly
    int_inst = ffi_arm32_le.new("int", -1)
    assert int_inst[0] == -1
    assert ffi_arm32_le.to_bytes(int_inst) == b"\xff\xff\xff\xff"

    # If the user sets it directly to a raw underflow, Python's int handles it
    # but let's test assignment wrap around
    int_inst[0] = 0xFFFFFFFF
    assert int_inst[0] == -1  # Because 'int' is marked signed in the ISF

    # Note: If we added an 'unsigned int' to the ISF, 0xFFFFFFFF would stay positive.

def test_endian_migration(ffi_arm32_le, ffi_mips32_be):
    """Test reading from a Little Endian struct and writing to a Big Endian one."""
    # LE Instance (ARM)
    le_task = ffi_arm32_le.new("struct cpu_context", {"pc": 0x11223344, "status": 0x55})
    le_bytes = bytes(le_task)
    assert le_bytes[0] == 0x44 # LE: LSB first
    
    # BE Instance (MIPS)
    be_task = ffi_mips32_be.new("struct cpu_context")
    
    # Manually migrate values
    be_task.pc = le_task.pc
    be_task.status = le_task.status
    
    be_bytes = bytes(be_task)
    assert be_bytes[0] == 0x11 # BE: MSB first
    
    # Values should match despite underlying byte differences
    assert be_task.pc == 0x11223344
    assert be_task.status == 0x55