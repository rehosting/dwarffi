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

@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_function_signatures(compiler):
    """Tests cross-compiler extraction of function arguments and return types."""
    ffi = DFFI()
    
    # We must provide function bodies, otherwise the compiler flags them as declarations
    # and our custom dwarf2json filter will drop them.
    ffi.cdef(
        """
        struct dummy_task {
            int id;
        };

        int do_math(int a, float b) {
            return a + (int)b;
        }

        void process_task(struct dummy_task *task, unsigned int flags) {
            if (task) {
                task->id = flags;
            }
        }

        struct dummy_task* get_current_task(void) {
            return (struct dummy_task*)0;
        }
        """,
        compiler=compiler
    )
    
    if not ffi.functions:
        pytest.skip("System dwarf2json does not support custom function signatures. Skipping.")

    # 1. Test Base Types
    do_math = ffi.get_function("do_math")
    assert do_math is not None
    
    # return_type is now a native VtypeBaseType! We can pass it straight to sizeof.
    assert ffi.sizeof(do_math.return_type) == 4
    
    assert len(do_math.args) == 2
    assert do_math.args[0].name == "a"
    assert ffi.sizeof(do_math.args[0].type) == 4
    
    # You still have direct access to the raw dict if ever needed
    assert do_math.args[1].type_info["name"] == "float"

    # 2. Test Struct Pointers and Void Returns
    process_task = ffi.get_function("process_task")
    assert process_task is not None
    assert process_task.return_type_info["kind"] == "base"
    assert len(process_task.args) == 2
    assert process_task.args[0].name == "task"
    
    # Pointer types resolve out to DFFI's standard pointer dictionaries
    assert process_task.args[0].type["kind"] == "pointer"
    assert process_task.args[0].type["subtype"]["name"] == "dummy_task"
    
    assert process_task.args[1].name == "flags"
    assert ffi.sizeof(process_task.args[1].type) == 4

@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_deep_function_signature_resolution(compiler):
    """
    Tests deeply nested type resolution (typedefs, enums, function pointers, 
    and pass-by-value structs) natively bridged through the function signature API.
    """
    ffi = DFFI()
    ffi.cdef(
        """
        typedef unsigned char u8;
        typedef u8 mac_addr_t[6];

        struct packet_info {
            mac_addr_t src;
            mac_addr_t dst;
            unsigned short length;
        };

        enum action_status {
            ACCEPT = 0,
            DROP = 1,
            ERROR = -1
        };

        // Typedef the enum
        typedef enum action_status action_t;
        
        // Typedef a function pointer that takes our struct
        typedef action_t (*filter_cb)(struct packet_info *pkt, void *user_data);

        struct filter_rule {
            filter_cb callback;
            void *user_data;
            unsigned int priority;
        };

        // The master function we are analyzing
        action_t register_filter(struct filter_rule *rule, filter_cb fallback, struct packet_info default_pkt) {
            if (rule && rule->callback) {
                return rule->callback(&default_pkt, rule->user_data);
            }
            if (fallback) {
                return fallback(&default_pkt, (void*)0);
            }
            return ERROR;
        }
        """,
        compiler=compiler
    )
    
    if not ffi.functions:
        pytest.skip("System dwarf2json does not support custom function signatures. Skipping.")

    # Grab the function wrapper
    reg_func = ffi.get_function("register_filter")
    assert reg_func is not None
    assert len(reg_func.args) == 3
    
    # --- 1. Return Type (Enum Typedef) ---
    # Because of our .bind() hook, this instantly resolves the typedef down to the native VtypeEnum!
    ret_type = reg_func.return_type
    assert ret_type.kind == "enum"
    assert ret_type.constants["ERROR"] == -1
    assert ret_type.constants["ACCEPT"] == 0
    assert ffi.sizeof(ret_type) == 4
    
    # --- 2. Arg 0: Pointer to Struct ---
    arg_rule = reg_func.args[0]
    assert arg_rule.name == "rule"
    
    # We can inspect the raw dictionary for the pointer type...
    assert arg_rule.type_info["kind"] == "pointer"
    
    # ...and easily resolve the underlying struct it points to using DFFI
    rule_struct = ffi.typeof(arg_rule.type_info["subtype"])
    assert rule_struct.name == "filter_rule"
    assert "callback" in rule_struct.fields
    assert "priority" in rule_struct.fields
    
    # --- 3. Arg 1: Function Pointer Typedef ---
    arg_cb = reg_func.args[1]
    assert arg_cb.name == "fallback"
    
    # Because it is a pointer, its size dynamically matches the architecture (4 or 8 bytes)
    ptr_size = ffi.sizeof("void *")
    assert ffi.sizeof(arg_cb.type) == ptr_size
    
    # --- 4. Arg 2: Pass-by-Value Struct ---
    arg_pkt = reg_func.args[2]
    assert arg_pkt.name == "default_pkt"
    
    # Since it was passed by value (not a pointer), .type directly yields the VtypeUserType
    pkt_struct = arg_pkt.type 
    assert pkt_struct.kind == "struct"
    assert pkt_struct.name == "packet_info"
    
    # Validate the size calculation recursively handles the inner mac_addr_t arrays
    # 6 bytes (src) + 6 bytes (dst) + 2 bytes (short) = 14 bytes
    assert ffi.sizeof(pkt_struct) == 14

@pytest.mark.parametrize("compiler", AVAILABLE_COMPILERS)
def test_e2e_struct_function_pointer_member(compiler):
    """
    Tests dynamic extraction of function signatures from members of a struct
    using a simulated file_operations jump table.
    """
    ffi = DFFI()
    ffi.cdef(
        """
        typedef long long loff_t;
        struct file {
            loff_t f_pos;
        };

        struct file_operations {
            loff_t (*llseek) (struct file *, loff_t, int);
            long (*unlocked_ioctl) (struct file *, unsigned int, unsigned long);
        };

        void __attribute__((used)) _force_keep(struct file_operations f) {}
        """,
        compiler=compiler
    )

    if not ffi.functions and not ffi.types.get("file_operations"):
        pytest.skip("System dwarf2json does not support custom function signatures or type resolution failed. Skipping.")

    inst = ffi.new("struct file_operations")

    # 1. Inspect llseek signature
    llseek_ptr = inst.llseek
    assert llseek_ptr.points_to_type_name == "function"

    sig1 = llseek_ptr.signature
    if not sig1:
        pytest.skip("dwarf2json version does not output function signatures for pointers. Skipping.")

    # Check if this dwarf2json version outputs full signature details
    if not sig1.args:
        pytest.skip("dwarf2json version does not output function signature parameters for pointers. Skipping.")

    assert sig1.return_type_info.get("name") == "loff_t"
    assert len(sig1.args) == 3

    # Arg 0: struct file *
    assert sig1.args[0].type_info["kind"] == "pointer"
    assert sig1.args[0].type_info["subtype"]["name"] == "file"

    # Arg 1: loff_t
    assert sig1.args[1].type_info["name"] == "loff_t"

    # Arg 2: int
    assert sig1.args[2].type_info["name"] == "int"

    # 2. Inspect unlocked_ioctl signature
    ioctl_ptr = inst.unlocked_ioctl
    sig2 = ioctl_ptr.signature
    assert sig2 is not None
    assert sig2.return_type_info.get("name") == "long"
    assert len(sig2.args) == 3
    assert sig2.args[1].type_info["name"] == "unsigned int"
    assert sig2.args[2].type_info["name"] == "unsigned long"