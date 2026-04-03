from dwarffi.dffi import DFFI
from dwarffi.types import VtypeBaseType, VtypeEnum, VtypeUserType


def test_exhaustive_typedef_coverage():
    d = DFFI()
    d.cdef("""
        // 1. Primitive Typedefs
        typedef unsigned char uint8_t;
        typedef unsigned long long uint64_t;
        
        // 2. Chained Primitive Typedefs
        typedef uint64_t size_t;
        
        // 3. Struct Typedefs (Named)
        struct _node { int data; };
        typedef struct _node node_t;
        
        // 4. Struct Typedefs (Anonymous)
        typedef struct { float x, y, z; } vector3_t;
        
        // 5. Array Typedefs
        typedef int matrix4x4_t[4][4];
        
        // 6. Pointer Typedefs
        typedef node_t* node_ptr_t;
        
        // 7. Enum Typedefs (Anonymous)
        typedef enum { STATUS_OK = 0, STATUS_ERR = -1 } status_t;
        
        // FORCE EMIT: A dummy function that uses all of our types as parameters.
        // Function signatures are tightly preserved in DWARF, ensuring the 
        // compiler does not optimize away the typedefs even at high optimization levels.
        // FORCE EMIT: A dummy function that uses all of our types as parameters.
        // We explicitly use 'struct _node' to prevent GCC from collapsing the typedef.
        void __attribute__((used)) _force_keep_func(
            uint8_t a,
            struct _node b,  // <-- Added to force struct _node into DWARF
            size_t c,
            node_t d,        // <-- Now GCC is forced to emit this as a typedef!
            vector3_t e,
            matrix4x4_t f,
            node_ptr_t g,
            status_t h
        ) {}
    """)

    typedefs_dict = d.vtypejsons[d._file_order[0]]._isf.typedefs

    # --- A. Verify the dictionary population from dwarf2json ---
    assert "uint8_t" in typedefs_dict
    assert "uint64_t" in typedefs_dict
    assert "size_t" in typedefs_dict
    print("Typedefs in ISF:", list(typedefs_dict.keys()))
    assert "node_t" in typedefs_dict
    assert "matrix4x4_t" in typedefs_dict
    assert "node_ptr_t" in typedefs_dict
    assert "status_t" in typedefs_dict

    # IMPORTANT: Anonymous structs are natively promoted to user_types by dwarf2json, 
    # so they bypass the typedefs dictionary completely!
    assert "vector3_t" not in typedefs_dict
    assert "vector3_t" in d.types

    # --- B. Verify get_type() resolution ---
    # With our patch, get_type cleanly unwraps the typedef and returns the true Vtype
    
    # Primitive
    t_u8 = d.get_type("uint8_t")
    assert isinstance(t_u8, VtypeBaseType)
    assert t_u8.name == "unsigned char"
    
    # Chained
    t_size = d.get_type("size_t")
    assert isinstance(t_size, VtypeBaseType)
    assert "long long" in t_size.name
    
    # Named Struct Typedef
    t_node = d.get_type("node_t")
    assert isinstance(t_node, VtypeUserType)
    assert t_node.name == "_node"
    
    # Anonymous Enum Typedef
    t_status = d.get_type("status_t")
    assert isinstance(t_status, VtypeEnum)

    # --- C. Verify typeof() for Dictionary-Based Types ---
    # get_type() cannot return pointers/arrays (they are ISF dicts, not Vtypes), 
    # so we verify typeof() catches them properly.
    
    t_mat = d.typeof("matrix4x4_t")
    assert isinstance(t_mat, dict)
    assert t_mat["kind"] == "array"
    
    t_ptr = d.typeof("node_ptr_t")
    assert isinstance(t_ptr, dict)
    assert t_ptr["kind"] == "pointer"

    # --- D. Verify the d.t Sugar Instantiation still works ---
    assert d.t.uint8_t(200) == 200
    assert d.t.node_t(data=42).data == 42
    assert d.t.status_t(0).name == "STATUS_OK"
    assert len(d.t.matrix4x4_t()) == 4
    assert d.t.node_ptr_t(0x1000).address == 0x1000


def test_chained_primitive_typedefs():
    """Tests that deeply chained typedefs resolve correctly to their base types."""
    d = DFFI()
    d.cdef("""
        typedef unsigned int u32;
        typedef u32 dword;
        typedef dword magic_t;
        
        // Force emission
        void __attribute__((used)) _force_keep(magic_t m) {}
    """)
    
    # 1. Type resolution
    t = d.get_type("magic_t")
    assert isinstance(t, VtypeBaseType)
    assert t.size == 4
    
    # 2. Instantiation
    inst = d.new("magic_t", 0xDEADBEEF)
    assert inst[0] == 0xDEADBEEF
    assert d.sizeof("magic_t") == 4
    
    # 3. Dynamic array view over a chained typedef
    arr = d.new("magic_t[10]")
    assert len(arr) == 10
    arr[0] = 0x1234
    assert arr[0] == 0x1234

def test_typedef_struct_and_arrays():
    """Tests typedefs that alias arrays, and typedefs embedded inside structs."""
    d = DFFI()
    d.cdef("""
        typedef float f32;
        typedef f32 vec4_t[4]; // Array typedef
        
        typedef struct {
            vec4_t position;
            vec4_t velocity;
            f32 mass;
        } physics_body_t;
        
        void __attribute__((used)) _force_keep(physics_body_t p) {}
    """)
    
    # 1. Typedef Array resolution
    arr_type = d.typeof("vec4_t")
    assert isinstance(arr_type, dict)
    assert arr_type["kind"] == "array"
    assert arr_type["count"] == 4
    assert d.sizeof("vec4_t") == 16  # 4 floats * 4 bytes
    
    # 2. Struct containing typedefs
    t = d.get_type("physics_body_t")
    assert isinstance(t, VtypeUserType)
    
    # 3. Deep Instantiation
    body = d.new("physics_body_t")
    body.position[0] = 1.5
    # Use 9.5 instead of 9.8 to avoid standard 32-bit floating point precision loss during assertions
    body.velocity[3] = 9.5 
    body.mass = 50.0
    
    assert body.position[0] == 1.5
    assert body.velocity[3] == 9.5
    assert d.sizeof("physics_body_t") == 36  # 16 + 16 + 4

def test_anonymous_enum_typedef():
    """Tests typedefs that alias completely anonymous enums."""
    d = DFFI()
    d.cdef("""
        typedef enum {
            STATE_INIT = 0,
            STATE_RUNNING = 1,
            STATE_STOPPED = 2
        } process_state_t;
        
        void __attribute__((used)) _force_keep(process_state_t s) {}
    """)
    
    # 1. Type resolution
    t = d.get_type("process_state_t")
    assert isinstance(t, VtypeEnum)
    
    # 2. Assignment by Integer Value
    state = d.new("process_state_t", 1)
    
    assert int(state[0]) == 1
    assert int(state) == 1
    
    # 3. Serialization extracts the raw primitive integer
    assert d.to_dict(state) == 1
    
    # 4. Stringification correctly resolves the enum's name
    assert d.string(state) == b"STATE_RUNNING"

def test_typedef_pointers_and_casting():
    """Tests typedefs that alias pointers, including double pointers."""
    d = DFFI()
    d.cdef("""
        struct node {
            int value;
            struct node* next;
        };
        typedef struct node* node_ptr_t;
        typedef node_ptr_t* node_ptr_ptr_t;
        
        void __attribute__((used)) _force_keep(node_ptr_ptr_t n) {}
    """)
    
    # 1. Pointer typedef resolution
    t1 = d.typeof("node_ptr_t")
    assert isinstance(t1, dict)
    assert t1["kind"] == "pointer"
    
    t2 = d.typeof("node_ptr_ptr_t")
    assert isinstance(t2, dict)
    assert t2["kind"] == "pointer"
    
    # Depending on the compiler/optimization, the inner pointer type may 
    # either preserve the typedef or decay directly into another pointer.
    assert t2["subtype"]["kind"] in ("typedef", "pointer")
    
    # 2. Casting integers to Typedef Pointers
    p = d.cast("node_ptr_t", 0x1000)
    assert p.address == 0x1000
    
    # 3. Nested Addressof casting
    inst = d.new("struct node")
    inst.value = 42
    
    # Get address of inst, cast it to our typedef pointer
    ptr_to_inst = d.cast("node_ptr_t", d.addressof(inst))
    
    # Verify the pointer cast preserved the address and updated the type natively
    # (We don't deref() here because inst was allocated in local Python memory, 
    # not the engine's backend)
    assert ptr_to_inst.address == d.addressof(inst).address
    assert ptr_to_inst.points_to_type_info["name"] == "node"

def test_typedef_shadowing():
    """Tests the C-language namespace rules where a typedef shares a name with a struct."""
    d = DFFI()
    d.cdef("""
        struct list_head {
            struct list_head *next, *prev;
        };
        
        // Typedef shares the exact same name as the struct
        typedef struct list_head list_head;
        
        void __attribute__((used)) _force_keep(list_head l) {}
    """)
    
    # Requesting "list_head" should route through the typedef safely
    t = d.get_type("list_head")
    assert isinstance(t, VtypeUserType)
    assert t.kind == "struct"
    assert t.name == "list_head"
    
    # Make sure we can allocate it
    head = d.new("list_head")
    print(head)
    assert d.sizeof("list_head") == 16  # Assuming 64-bit pointers

def test_function_pointer_typedef():
    """Tests typedefs wrapping function pointers."""
    d = DFFI()
    d.cdef("""
        typedef int (*math_op_t)(int, int);
        
        struct math_context {
            math_op_t op;
            int last_result;
        };
        
        void __attribute__((used)) _force_keep(struct math_context m) {}
    """)
    
    t = d.typeof("math_op_t")
    assert isinstance(t, dict)
    assert t["kind"] == "pointer"
    
    ctx = d.new("struct math_context")
    ctx.op = 0x400500 # Assign an arbitrary memory address representing a function
    assert ctx.op == 0x400500

def test_typedef_unions():
    """Tests typedefs that alias unions."""
    d = DFFI()
    d.cdef("""
        typedef union {
            unsigned int raw;
            unsigned char bytes[4];
        } color_t;
        
        void __attribute__((used)) _force_keep(color_t c) {}
    """)
    
    # 1. Type resolution
    t = d.get_type("color_t")
    assert isinstance(t, VtypeUserType)
    assert t.kind == "union"
    
    # 2. Assignment and overlapping memory verification
    color = d.new("color_t")
    color.raw = 0xFF00AA55
    
    # Depending on endianness, the bytes will overlap. 
    # Assuming Little Endian (standard x86/ARM):
    assert color.bytes[0] == 0x55
    assert color.bytes[1] == 0xAA
    assert color.bytes[2] == 0x00
    assert color.bytes[3] == 0xFF
    assert d.sizeof("color_t") == 4

def test_opaque_handle_typedefs():
    """Tests 'opaque pointers' (pointers to incomplete structs) commonly used for handles."""
    d = DFFI()
    d.cdef("""
        // The struct is never fully defined in this compilation unit
        struct _internal_state; 
        
        // But we typedef a pointer to it (like HWND or FILE*)
        typedef struct _internal_state* HANDLE;
        
        void __attribute__((used)) _force_keep(HANDLE h) {}
    """)
    
    t = d.typeof("HANDLE")
    assert isinstance(t, dict)
    assert t["kind"] == "pointer"
    
    # It should point to a struct named '_internal_state' even if the size is unknown
    assert t["subtype"]["name"] == "_internal_state"
    assert t["subtype"]["kind"] == "struct"
    
    # We should still be able to cast and pass around the handles as raw integers
    handle = d.cast("HANDLE", 0x8BADF00D)
    assert handle.address == 0x8BADF00D

def test_multidimensional_array_typedef_chains():
    """Tests typedefs that build multi-dimensional arrays out of 1D array typedefs."""
    d = DFFI()
    d.cdef("""
        typedef float vec3_t[3];
        typedef vec3_t transform_matrix_t[4]; // Array of arrays
        
        void __attribute__((used)) _force_keep(transform_matrix_t t) {}
    """)
    
    t = d.typeof("transform_matrix_t")
    assert isinstance(t, dict)
    assert t["kind"] == "array"
    assert t["count"] == 4
    
    # The subtype should resolve to the inner typedef or decay to its array equivalent
    subtype = d._resolve_type_info(t["subtype"])
    assert subtype["kind"] == "array"
    assert subtype["count"] == 3
    
    # Matrix size = 4 rows * 3 cols * 4 bytes = 48 bytes
    assert d.sizeof("transform_matrix_t") == 48
    
    # Test instantiation and 2D access
    matrix = d.new("transform_matrix_t")
    
    # Use 3.5 instead of 3.14 to avoid 32-bit float precision loss during assertion
    matrix[1][2] = 3.5 
    
    assert matrix[1][2] == 3.5

def test_self_referential_typedef_structs():
    """Tests typedefs of structs that contain pointers to their own type (e.g., Tree Nodes)."""
    d = DFFI()
    d.cdef("""
        typedef struct _tree_node {
            int id;
            struct _tree_node* left;
            struct _tree_node* right;
        } tree_node_t;
        
        typedef tree_node_t* tree_ptr_t;
        
        void __attribute__((used)) _force_keep(tree_ptr_t t) {}
    """)
    
    t = d.get_type("tree_node_t")
    assert isinstance(t, VtypeUserType)
    
    # Allocate a root and two children
    root = d.new("tree_node_t")
    child_l = d.new("tree_node_t")
    child_r = d.new("tree_node_t")
    
    root.id = 100
    child_l.id = 50
    child_r.id = 150
    
    # Link them using addressof (creates Ptr objects)
    root.left = d.addressof(child_l)
    root.right = d.addressof(child_r)
    
    # Verify the pointers hold the correct absolute addresses
    assert root.left.address == d.addressof(child_l).address
    assert root.right.address == d.addressof(child_r).address
    
    # Verify the pointer's inner type correctly resolves to the self-referential struct
    assert root.left.points_to_type_info["name"] == "_tree_node"
    assert root.right.points_to_type_info["name"] == "_tree_node"

def test_hardware_register_bitfields_with_typedefs():
    """Tests deeply nested anonymous structs, unions, and bitfields built entirely out of typedefs."""
    d = DFFI()
    d.cdef("""
        typedef unsigned char u8;
        typedef unsigned short u16;
        typedef unsigned int u32;

        typedef union {
            u32 raw;
            struct {
                u8 is_enabled : 1;
                u8 has_error  : 1;
                u8 reserved   : 6;
                u8 data_code;
                u16 memory_page;
            } fields;
        } hw_register_t;
        
        void __attribute__((used)) _force_keep(hw_register_t r) {}
    """)
    
    reg = d.new("hw_register_t")
    
    # Size should be exactly 4 bytes (32 bits)
    assert d.sizeof("hw_register_t") == 4
    
    # Write to the bitfields natively
    reg.fields.is_enabled = 1
    reg.fields.has_error = 0
    reg.fields.data_code = 0xAA
    reg.fields.memory_page = 0xBEEF
    
    # The raw union overlap should perfectly reflect the packed memory!
    # Bit layout (Little Endian):
    # Byte 0: [ reserved:6 | has_error:1 | is_enabled:1 ] -> 00000001 = 0x01
    # Byte 1: data_code = 0xAA
    # Byte 2-3: memory_page = 0xBEEF
    # Total expected raw u32: 0xBEEFAA01
    
    assert reg.raw == 0xBEEFAA01
    
    # Reverse test: write to raw, read from bitfields
    reg.raw = 0x12345603 # 0x03 -> is_enabled=1, has_error=1
    assert reg.fields.is_enabled == 1
    assert reg.fields.has_error == 1
    assert reg.fields.data_code == 0x56
    assert reg.fields.memory_page == 0x1234

def test_vtable_simulation_typedefs():
    """Tests structs containing arrays of function pointers (vtables) via typedefs."""
    d = DFFI()
    d.cdef("""
        // Typedef for a function pointer that takes an int and returns void
        typedef void (*action_func_t)(int);
        
        // Typedef for a struct containing an array of those function pointers
        typedef struct {
            int state;
            action_func_t vtable[4];
        } object_class_t;
        
        void __attribute__((used)) _force_keep(object_class_t o) {}
    """)
    
    obj = d.new("object_class_t")
    
    # Set the state
    obj.state = 404
    
    # Assign some fake memory addresses representing loaded functions to the vtable
    obj.vtable[0] = 0x08048000
    obj.vtable[3] = 0x080480F0
    
    # Read them back out
    assert obj.state == 404
    assert obj.vtable[0] == 0x08048000
    assert obj.vtable[1] == 0  # Uninitialized should be 0
    assert obj.vtable[3] == 0x080480F0
    
    # Verify exact size (4 bytes for int + 4 * pointer_size)
    ptr_size = d.sizeof("pointer")
    expected_size = 4 + (4 * ptr_size)
    
    # Account for C-struct padding: if pointers are 8 bytes, the 'int' will have 4 bytes of padding after it
    if ptr_size == 8:
        expected_size += 4 
        
    assert d.sizeof("object_class_t") == expected_size