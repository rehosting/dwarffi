import shutil
import pytest

from dwarffi import DFFI

# ---------------------------------------------------------------------------
# Setup for C++ E2E Compiler Tests
# ---------------------------------------------------------------------------
CPP_COMPILERS = {
    "g++": shutil.which("g++"),
    "clang++": shutil.which("clang++"),
}

AVAILABLE_CPP_COMPILERS = [path for name, path in CPP_COMPILERS.items() if path is not None]
HAS_DWARF2JSON = shutil.which("dwarf2json") is not None

pytestmark = pytest.mark.skipif(
    not HAS_DWARF2JSON or not AVAILABLE_CPP_COMPILERS, 
    reason="dwarf2json or C++ compilers missing from PATH"
)

@pytest.mark.parametrize("compiler", AVAILABLE_CPP_COMPILERS)
def test_e2e_cpp_pointer_to_member(compiler):
    """
    Tests that a C++ pointer-to-member function is accurately sized and accessible.
    In the Itanium ABI (used by Linux), these are 16 bytes (pointer + adjustment).
    """
    ffi = DFFI()
    ffi.cdef(
        """
        struct CppClass {
            int normal_int;
            void (CppClass::*member_func_ptr)();
        };
        // Force compiler to emit debug symbols for the class
        CppClass _force_emission;
        """,
        compiler=f"{compiler} -x c++"
    )
    
    try:
        inst = ffi.new("CppClass")
    except KeyError:
        inst = ffi.new("struct CppClass")
        
    assert ffi.sizeof(inst._instance_type_name) >= 20
    
    inst.normal_int = 42
    assert inst.normal_int == 42


@pytest.mark.parametrize("compiler", AVAILABLE_CPP_COMPILERS)
def test_e2e_cpp_vtable_injection(compiler):
    """
    Tests that C++ classes with virtual functions have the correct size and layout
    accounting for the hidden vtable pointer injected by the compiler.
    """
    ffi = DFFI()
    ffi.cdef(
        """
        struct VirtualClass {
            virtual void do_something() {}
            int user_data;
        };
        // Force compiler to emit debug symbols including the vtable
        VirtualClass _force_emission;
        """,
        compiler=f"{compiler} -x c++"
    )
    
    try:
        inst = ffi.new("VirtualClass")
    except KeyError:
        inst = ffi.new("struct VirtualClass")
        
    inst.user_data = 1337
    assert inst.user_data == 1337
    
    assert ffi.sizeof(inst._instance_type_name) == 16
    
    fields = inst._instance_type_def.fields
    vptr_fields = [f for f in fields if "vptr" in f]
    
    assert len(vptr_fields) == 1


@pytest.mark.parametrize("compiler", AVAILABLE_CPP_COMPILERS)
def test_e2e_cpp_references(compiler):
    """
    Tests C++ references (&). Under the hood, the compiler implements these as
    standard pointers, and DWARF tags them with DW_TAG_reference_type.
    """
    ffi = DFFI()
    ffi.cdef(
        """
        struct RefHolder {
            int& my_ref;
        };
        
        // Force emission of 'pointer' base type in DWARF
        void* _force_ptr_emission;
        RefHolder* _force_class_emission;
        """,
        compiler=f"{compiler} -x c++"
    )
    
    try:
        inst = ffi.new("RefHolder")
    except KeyError:
        inst = ffi.new("struct RefHolder")
        
    assert ffi.sizeof(inst._instance_type_name) == ffi.sizeof("void *")