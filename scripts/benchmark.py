import struct
import time

from dwarffi.dffi import DFFI

# ---------------------------------------------------------
# 1. Define a realistic dummy ISF for the workloads
# ---------------------------------------------------------
MOCK_ISF = {
    "metadata": {},
    "base_types": {
        "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
        "long": {"kind": "int", "size": 8, "signed": True, "endian": "little"},
        "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
        "byte": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
    },
    "user_types": {
        # Scenario 1: Standard Linux-style intrusive linked list node
        "list_head": {
            "kind": "struct", "size": 16,
            "fields": {
                "next": {"offset": 0, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "list_head"}}},
                "prev": {"offset": 8, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "list_head"}}}
            }
        },
        "task_node": {
            "kind": "struct", "size": 32,
            "fields": {
                "pid": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                "state": {"offset": 4, "type": {"kind": "base", "name": "int"}},
                "node": {"offset": 8, "type": {"kind": "struct", "name": "list_head"}},
                "flags": {"offset": 24, "type": {"kind": "base", "name": "long"}}
            }
        },
        # Scenario 2: Network / IPC Ring Buffer entry
        "ring_entry": {
            "kind": "struct", "size": 64,
            "fields": {
                "timestamp": {"offset": 0, "type": {"kind": "base", "name": "long"}},
                "size": {"offset": 8, "type": {"kind": "base", "name": "int"}},
                "magic": {"offset": 12, "type": {"kind": "base", "name": "int"}},
                "data": {"offset": 16, "type": {"kind": "array", "count": 48, "subtype": {"kind": "base", "name": "byte"}}}
            }
        },
        # Scenario 3: Hardware Page Table Entry (Bitfields)
        "pte_struct": {
            "kind": "struct", "size": 8,
            "fields": {
                "present": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 1, "bit_position": 0, "type": {"name": "long"}}},
                "rw": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 1, "bit_position": 1, "type": {"name": "long"}}},
                "user": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 1, "bit_position": 2, "type": {"name": "long"}}},
                "pfn": {"offset": 0, "type": {"kind": "bitfield", "bit_length": 52, "bit_position": 12, "type": {"name": "long"}}}
            }
        }
    },
    "enums": {}, "symbols": {}, "typedefs": {}
}

# ---------------------------------------------------------
# Workload 1: Deep Linked List Traversal
# Stresses: addressof, Ptr dereferencing, nested struct field access
# ---------------------------------------------------------
def bench_linked_list_walk(num_nodes: int = 250_000):
    print(f"--- Running Linked List Walk ({num_nodes} nodes) ---")
    
    # Boot a temporary engine just to calculate sizes
    temp_ffi = DFFI(MOCK_ISF)
    node_size = temp_ffi.sizeof("struct task_node")
    
    # Create the global memory space
    buf = bytearray(node_size * num_nodes)
    
    # Wire up the intrusive linked list directly in the buffer
    for i in range(num_nodes):
        current_offset = i * node_size
        next_offset = ((i + 1) % num_nodes) * node_size
        
        # Write PID
        struct.pack_into("<i", buf, current_offset, i + 1000)
        # Write list_head.next pointer (address is absolute offset here)
        struct.pack_into("<Q", buf, current_offset + 8, next_offset + 8)

    # Boot the REAL engine with the buffer attached as the Memory Backend
    ffi = DFFI(MOCK_ISF, backend=buf)

    # Begin the walk from address 0
    root = ffi.from_address("struct task_node", 0)
    
    # We want a pointer to the first list_head
    current_ptr = ffi.addressof(root, "node")
    
    start_t = time.perf_counter()
    
    total_pids = 0
    # Walk the list using container_of style offset math
    list_head_offset = ffi.offsetof("struct task_node", "node")
    
    for _ in range(num_nodes):
        # Dereference pointer to list_head (requires a backend!)
        list_node = current_ptr.deref()
        
        # Calculate parent task_node address (simulate container_of)
        task_addr = current_ptr.address - list_head_offset
        task = ffi.from_address("struct task_node", task_addr)
        
        total_pids += task.pid
        current_ptr = list_node.next
        
    end_t = time.perf_counter()
    print(f"Done in {end_t - start_t:.4f}s (Sum: {total_pids})\n")

# ---------------------------------------------------------
# Workload 2: High-Throughput Array iteration
# Stresses: BoundArrayView, __getitem__, primitive base types
# ---------------------------------------------------------
def bench_ring_buffer_scan(ffi: DFFI, num_entries: int = 500_000):
    print(f"--- Running Ring Buffer Array Scan ({num_entries} entries) ---")
    entry_size = ffi.sizeof("struct ring_entry")
    buf = bytearray(entry_size * num_entries)
    
    # Pre-seed some dummy data using native struct for speed
    for i in range(num_entries):
        struct.pack_into("<qii", buf, i * entry_size, 1600000000 + i, 42, 0x7EADBEEF)

    # Create the array view
    ring = ffi.from_buffer(f"struct ring_entry[{num_entries}]", buf)
    
    start_t = time.perf_counter()
    
    valid_packets = 0
    total_payload_size = 0
    
    # Iterate using python loops over the C-Array view
    for i in range(num_entries):
        entry = ring[i]
        if entry.magic == 0xDEADBEEF:
            valid_packets += 1
            total_payload_size += entry.size
            
    end_t = time.perf_counter()
    print(f"Done in {end_t - start_t:.4f}s (Valid: {valid_packets}, Total Data: {total_payload_size})\n")

# ---------------------------------------------------------
# Workload 3: Bitfield Thrashing
# Stresses: __setattr__, bitfield packing/unpacking math, wrapping
# ---------------------------------------------------------
def bench_bitfield_mutations(ffi: DFFI, iterations: int = 300_000):
    print(f"--- Running Bitfield Thrashing ({iterations} loops) ---")
    buf = bytearray(8)
    pte = ffi.from_buffer("struct pte_struct", buf)
    
    start_t = time.perf_counter()
    
    for i in range(iterations):
        pte.present = 1
        pte.rw = i % 2
        pte.user = 0
        pte.pfn = i
        
        # Read back to force the unpacker
        _ = pte.pfn
        
    end_t = time.perf_counter()
    print(f"Done in {end_t - start_t:.4f}s (Final PFN: {pte.pfn})\n")

if __name__ == "__main__":
    # We can use a backend-less FFI for scenarios 2 and 3, which just use from_buffer
    global_ffi = DFFI(MOCK_ISF)
    
    # Workload 1 manages its own backend now
    bench_linked_list_walk()
    bench_ring_buffer_scan(global_ffi)
    bench_bitfield_mutations(global_ffi)