from dwarffi.dffi import DFFI


def main():
    # 1. Load a pre-compiled ISF file (e.g., generated via your CI pipeline)
    # d = DFFI("vmlinux.json.xz")
    
    # For this example, we'll mock loading an ISF dictionary directly
    # since `d.cdef()` optimizes for types and explicitly omits symbol addresses.
    d = DFFI({
        "base_types": {
            "int": {"kind": "int", "size": 4}
        },
        "user_types": {
            "task_struct": {"kind": "struct", "size": 100, "fields": {}}
        },
        "symbols": {
            "system_state": {
                "address": 0xffffffff81000000, 
                "type": {"kind": "base", "name": "int"}
            },
            "init_task": {
                "address": 0xffffffff82000000, 
                "type": {"kind": "struct", "name": "task_struct"}
            }
        }
    })
    
    # 2. Accessing Global Symbols via d.sym
    # This automatically looks up the address of the symbol in the ELF/PDB
    print(f"Address of system_state: {hex(d.sym.system_state.address)}")
    print(f"Address of init_task: {hex(d.sym.init_task.address)}")
    
    # 3. Searching for Types and Symbols
    # Supports glob patterns to find things quickly
    tasks = d.search_types("*task*")
    print(f"Found task types: {list(tasks.keys())}")

if __name__ == "__main__":
    main()