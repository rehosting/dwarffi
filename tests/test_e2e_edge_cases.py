import shutil

import pytest

from dwarffi import DFFI

COMPILERS = {
    "gcc": shutil.which("gcc"),
    "clang": shutil.which("clang"),
    "arm32": shutil.which("arm-linux-gnueabi-gcc"),
    "aarch64": shutil.which("aarch64-linux-gnu-gcc"),
}

AVAILABLE_COMPILERS = [path for name, path in COMPILERS.items() if path is not None]
HAS_DWARF2JSON = shutil.which("dwarf2json") is not None

pytestmark = pytest.mark.skipif(
    not HAS_DWARF2JSON or not AVAILABLE_COMPILERS, 
    reason="dwarf2json or compilers missing from PATH"
)


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_zero_length_arrays(compiler):
    """Tests C99 flexible array members (zero-length arrays)."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct packet {
            int length;
            unsigned char payload[0]; 
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct packet")
    inst.length = 100
    
    # The array view should exist but explicitly know it has 0 bounds tracked
    assert len(inst.payload) == 0
    
    # We shouldn't be able to access indices, but C-style pointer decay should 
    # perfectly resolve the memory address immediately after the `length` field
    payload_ptr = inst.payload + 0
    assert payload_ptr.address == ffi.sizeof("int")


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_self_referential_and_opaque(compiler):
    """Tests pointers to undefined forward-declarations and self-referencing structs."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct opaque_type;
        
        struct list_node {
            int value;
            struct list_node *next;
            struct opaque_type *data;
        };
        """,
        compiler=compiler
    )
    
    node = ffi.new("struct list_node")
    node.value = 10
    
    # Assign arbitrary memory addresses to the pointers
    node.next = 0x1000
    node.data = 0x2000
    
    # Reading them back should yield Ptr objects that haven't crashed trying to resolve
    assert node.next.address == 0x1000
    assert "list_node" in node.next.points_to_type_name
    
    assert node.data.address == 0x2000
    # Opaque types might be omitted by DWARF or resolved as void/opaque depending on the compiler
    assert node.data.points_to_type_name in ["opaque_type", "void"] 


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_packed_structs(compiler):
    """Tests strictly packed structures to ensure unaligned memory access works."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct __attribute__((__packed__)) packed_data {
            unsigned char a;
            int b;
            short c;
        };
        """,
        compiler=compiler
    )
    
    # 1 byte (char) + 4 bytes (int) + 2 bytes (short) = Exactly 7 bytes (no padding)
    assert ffi.sizeof("struct packed_data") == 7
    
    inst = ffi.new("struct packed_data")
    
    # Write to an unaligned 4-byte integer (starts at byte offset 1)
    inst.a = 0xAA
    inst.b = 0x11223344
    inst.c = 0x5566
    
    assert inst.a == 0xAA
    assert inst.b == 0x11223344
    assert inst.c == 0x5566


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_multidimensional_arrays(compiler):
    """Tests deeply nested BoundArrayViews."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct grid {
            int matrix[3][4];
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct grid")
    
    # Accessing an inner array element dynamically chains offsets
    inst.matrix[2][3] = 42
    assert inst.matrix[2][3] == 42
    
    # Matrix is 3 elements, each is an array of 4 ints. (3 * 4 * 4 bytes = 48 bytes)
    assert ffi.sizeof("struct grid") == 48


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_modifiers(compiler):
    """Ensures const/volatile/restrict keywords do not break type parsing."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct hardware_regs {
            volatile unsigned int status;
            const unsigned int device_id;
            unsigned int * restrict buf_ptr;
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct hardware_regs")
    
    # Even though C says device_id is const, our memory viewer should bypass that 
    # and just write the bytes directly based on the underlying raw type.
    inst.status = 1
    inst.device_id = 0x1234
    inst.buf_ptr = 0xDEADBEEF
    
    assert inst.status == 1
    assert inst.device_id == 0x1234
    assert inst.buf_ptr.address == 0xDEADBEEF


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_zero_width_bitfields(compiler):
    """
    Tests zero-width bitfields.
    In C, `int : 0` tells the compiler to pad to the next alignment boundary
    for the specified type. It takes up no space itself but forces gaps.
    """
    ffi = DFFI()
    ffi.cdef(
        """
        struct zero_width_test {
            unsigned char a;
            int : 0;  // Force alignment to the next integer boundary
            unsigned char b;
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct zero_width_test")
    
    # We shouldn't be able to access the anonymous zero-width field,
    # but 'a' and 'b' should work and have a massive gap between them.
    inst.a = 0xAA
    inst.b = 0xBB
    
    assert inst.a == 0xAA
    assert inst.b == 0xBB
    
    # Verify the internal offset gap by checking the raw byte buffer
    # Offset 0 = a
    # Offset 1-3 = padding
    # Offset 4 = b
    raw_bytes = bytes(inst)
    assert raw_bytes[0] == 0xAA
    assert raw_bytes[4] == 0xBB
    
    # Size should be 5 on x86_64 (1 byte a + 3 bytes pad + 1 byte b, struct alignment = 1)
    # Could be 8 on strictly aligned architectures like some ARM variants
    assert ffi.sizeof("struct zero_width_test") in (5, 8)


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_forced_alignment(compiler):
    """Tests __attribute__((aligned(X))) which heavily distorts struct sizes."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct over_aligned {
            unsigned char tiny;
            long long huge __attribute__((aligned(64)));
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct over_aligned")
    inst.tiny = 0xFF
    inst.huge = 0x1122334455667788
    
    assert inst.tiny == 0xFF
    assert inst.huge == 0x1122334455667788
    
    # The struct size must be a multiple of the largest alignment requirement (64)
    # Offset 0: tiny (1 byte)
    # Offset 1-63: padding
    # Offset 64: huge (8 bytes)
    # Offset 72-127: tail padding
    assert ffi.sizeof("struct over_aligned") == 128


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_gnu_empty_structs(compiler):
    """
    Tests empty structures. Standard C says structs must have >0 members,
    but GNU C allows empty structs (which take 0 bytes in C, but 1 byte in C++).
    """
    ffi = DFFI()
    ffi.cdef(
        """
        struct empty {};
        struct container {
            int before;
            struct empty nothing;
            int after;
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct container")
    inst.before = 123
    inst.after = 456
    
    assert inst.before == 123
    assert inst.after == 456
    
    # In C, `nothing` takes exactly 0 bytes, so `after` immediately follows `before`
    assert ffi.sizeof("struct container") == 8
    
    # Trying to read the empty struct should safely return a BoundTypeInstance 
    # bounded to 0 bytes
    empty_inst = inst.nothing
    assert len(bytes(empty_inst)) == 0


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_array_of_anonymous_unions(compiler):
    """Tests the parser's ability to chain lookups through arrays of anonymous types."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct multi_instruction {
            unsigned char opcode;
            union {
                unsigned int immediate;
                struct {
                    unsigned short reg1;
                    unsigned short reg2;
                };
            } operands[4];
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct multi_instruction")
    
    # Set the immediate value of the 2nd operand
    inst.operands[1].immediate = 0xAABBCCDD
    
    # Read it back via the anonymous struct fields (assumes little-endian)
    assert inst.operands[1].reg1 == 0xCCDD
    assert inst.operands[1].reg2 == 0xAABB
    
    # Ensure other array elements were not corrupted
    assert inst.operands[0].immediate == 0
    assert inst.operands[2].immediate == 0


@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_function_returning_function_pointer(compiler):
    """Tests complex, deeply nested function pointer syntax (e.g., the signal() pattern)."""
    ffi = DFFI()
    ffi.cdef(
        """
        // sighandler_t is a pointer to a function taking int, returning void
        typedef void (*sighandler_t)(int);
        
        struct state_machine {
            // A function that returns a sighandler_t
            sighandler_t (*get_handler)(void);
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct state_machine")
    inst.get_handler = 0xCAFEBABE
    
    assert inst.get_handler.address == 0xCAFEBABE
    # The string representation should gracefully handle the nested type info
    assert "function" in inst.get_handler.points_to_type_name