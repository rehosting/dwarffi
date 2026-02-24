import cProfile
import pstats

from dwarffi.dffi import DFFI


def simulated_rehosting_workload():
    """Simulates a heavy workload to see where DFFI spends its time."""
    # 1. Setup a realistic kernel-like ISF
    isf = {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"},
        },
        "user_types": {
            "list_head": {
                "kind": "struct", "size": 16,
                "fields": {
                    "next": {"offset": 0, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "list_head"}}},
                    "prev": {"offset": 8, "type": {"kind": "pointer", "subtype": {"kind": "struct", "name": "list_head"}}},
                }
            },
            "task_struct": {
                "kind": "struct", "size": 24,
                "fields": {
                    "pid": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                    "tasks": {"offset": 8, "type": {"kind": "struct", "name": "list_head"}},
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    
    ffi = DFFI(isf)
    
    # 2. Simulate type resolution spam (happens a lot when casting)
    for _ in range(5000):
        ffi.typeof("struct task_struct *")
        ffi.sizeof("struct list_head")

    # 3. Simulate memory instantiation and field writes
    tasks = []
    for i in range(1000):
        t = ffi.new("struct task_struct")
        t.pid = i
        t.tasks.next = 0xFFFF0000 + (i * 24) # Dummy pointers
        t.tasks.prev = 0xFFFF0000 - (i * 24)
        tasks.append(t)
        
    # 4. Simulate a linked-list walk (heavy __getattr__ usage)
    for t in tasks:
        _ = t.pid
        _ = t.tasks.next
        _ = t.tasks.prev

if __name__ == "__main__":
    print("[*] Running macro workload with cProfile...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    simulated_rehosting_workload()
    
    profiler.disable()
    
    # Dump the stats to the console, sorted by cumulative time
    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    
    print("\n--- Top 20 Time-Consuming Functions ---")
    stats.print_stats(20)
    
    # Optional: Save for visualization
    stats.dump_stats("dwarffi_macro.prof")
    print("\n[+] Profiling data saved to 'dwarffi_macro.prof'")
    print("    Run 'snakeviz dwarffi_macro.prof' to view the flamegraph in your browser.")