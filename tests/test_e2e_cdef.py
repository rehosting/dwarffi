import os
import shutil
import pytest
from dwarffi import DFFI

COMPILERS = {
    "gcc": shutil.which("gcc"),
    "clang": shutil.which("clang"),
    "arm32": shutil.which("arm-linux-gnueabi-gcc"),
    "aarch64": shutil.which("aarch64-linux-gnu-gcc"),
    "mips": shutil.which("mips-linux-gnu-gcc"),
}

AVAILABLE_COMPILERS = [path for name, path in COMPILERS.items() if path is not None]
HAS_DWARF2JSON = shutil.which("dwarf2json") is not None

pytestmark = pytest.mark.skipif(
    not HAS_DWARF2JSON or not AVAILABLE_COMPILERS, 
    reason="dwarf2json or compilers missing from PATH"
)

@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_complex_bitfields(compiler):
    """Compiles a struct with tightly packed bitfields mixed with standard types."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct bitfield_edge {
            unsigned int f1 : 3;
            int f2 : 6;
            unsigned int f3 : 15;
            unsigned char flag;
        };
        """,
        compiler=compiler
    )
    
    inst = ffi.new("struct bitfield_edge")
    
    # Unsigned truncation
    inst.f1 = 7
    assert inst.f1 == 7
    inst.f1 = 8  # 0b1000 truncates to 0
    assert inst.f1 == 0
    
    # Signed boundaries
    inst.f2 = -15
    assert inst.f2 == -15
    inst.f2 = -32
    assert inst.f2 == -32
    
    # Multi-byte bitfield crossing boundaries
    inst.f3 = 32767
    assert inst.f3 == 32767
    
    inst.flag = 0xAA
    assert inst.flag == 0xAA

@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_deep_anonymous_unions(compiler):
    """Compiles C11 anonymous structures and unions typical in kernel headers."""
    ffi = DFFI()
    ffi.cdef(
        """
        struct hw_register {
            union {
                unsigned int raw32;
                struct {
                    unsigned short low16;
                    unsigned short high16;
                };
                struct {
                    unsigned char b0;
                    unsigned char b1;
                    unsigned char b2;
                    unsigned char b3;
                };
            };
        };
        """,
        compiler=compiler
    )
    
    reg = ffi.new("struct hw_register")
    reg.raw32 = 0xAABBCCDD
    
    # These asserts assume Little Endian, which is default for these cross-compilers on Linux
    assert reg.b0 == 0xDD
    assert reg.b1 == 0xCC
    assert reg.b2 == 0xBB
    assert reg.b3 == 0xAA
    
    assert reg.low16 == 0xCCDD
    assert reg.high16 == 0xAABB

@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_multidimensional_arrays_and_typedefs(compiler):
    """Tests resolving arrays of structs that contain arrays of typedefs."""
    ffi = DFFI()
    ffi.cdef(
        """
        typedef unsigned char u8;
        typedef u8 mac_addr_t[6];
        
        typedef struct {
            mac_addr_t src;
            mac_addr_t dst;
            unsigned short ethertype;
        } eth_header_t;
        
        struct packet_buffer {
            eth_header_t headers[4];
            unsigned int count;
        };
        """,
        compiler=compiler
    )
    
    pkt = ffi.new("struct packet_buffer")
    pkt.headers[1].src[5] = 0xEE
    pkt.headers[2].dst[0] = 0xFF
    pkt.headers[3].ethertype = 0x0800
    pkt.count = 4

    assert pkt.headers[1].src[5] == 0xEE
    assert pkt.headers[2].dst[0] == 0xFF
    assert pkt.headers[3].ethertype == 0x0800
    assert pkt.count == 4

@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_function_pointers_and_callbacks(compiler):
    """Tests function pointers and dynamic architecture pointer sizing."""
    ffi = DFFI()
    ffi.cdef(
        """
        typedef int (*callback_t)(void *ctx, int event);
        
        struct event_handler {
            callback_t on_event;
            void *context;
        };
        """,
        compiler=compiler
    )
    
    handler = ffi.new("struct event_handler")
    handler.on_event = 0xDEADBEEF
    
    ptr = handler.on_event
    assert ptr.address == 0xDEADBEEF
    
    ptr_size = ffi.sizeof("void *")
    struct_size = ffi.sizeof("struct event_handler")
    
    # Ensures size alignment matches the target architecture ABI (4 vs 8 bytes)
    assert struct_size == (ptr_size * 2)

@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_tricky_enums(compiler):
    """Tests enums containing negative numbers and standard max bounds."""
    ffi = DFFI()
    ffi.cdef(
        """
        enum status_codes {
            STATUS_OK = 0,
            STATUS_ERR_GENERIC = -1,
            STATUS_ERR_TIMEOUT = -2,
            STATUS_PENDING = 2147483647
        };
        struct result {
            enum status_codes code;
        };
        """,
        compiler=compiler
    )
    
    res = ffi.new("struct result")
    
    # Verify assignment logic can reverse the string correctly even on boundaries
    res.code = -1
    assert res.code.name == "STATUS_ERR_GENERIC"
    
    res.code = "STATUS_ERR_TIMEOUT"
    assert int(res.code) == -2