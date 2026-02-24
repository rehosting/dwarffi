import timeit

from dwarffi import DFFI


def run_profiler():
    print("[*] Setting up performance benchmark...")
    
    isf = {
        "metadata": {},
        "base_types": {
            "u64": {"kind": "int", "size": 8, "signed": False, "endian": "little"},
            "u32": {"kind": "int", "size": 4, "signed": False, "endian": "little"},
            "u16": {"kind": "int", "size": 2, "signed": False, "endian": "little"},
        },
        "user_types": {
            # A 10-field struct simulating a heavy kernel or hardware object
            "heavy_struct": {
                "kind": "struct", "size": 40,
                "fields": {
                    f"f{i}": {"offset": i * 4, "type": {"kind": "base", "name": "u32"}}
                    for i in range(10)
                }
            }
        },
        "enums": {}, "symbols": {}
    }
    
    ffi = DFFI(isf)
    inst = ffi.new("struct heavy_struct")
    
    # Initialize with some dummy data
    for i in range(10):
        setattr(inst, f"f{i}", i * 100)

    # The two functions we are racing
    def method_field_by_field():
        return (inst.f0, inst.f1, inst.f2, inst.f3, inst.f4, 
                inst.f5, inst.f6, inst.f7, inst.f8, inst.f9)

    def method_bulk_unpack():
        return ffi.unpack(inst)

    # Warmup and verify
    assert method_field_by_field() == method_bulk_unpack(), "Mismatch in extracted data!"

    ITERATIONS = 500_000
    print(f"[*] Running {ITERATIONS:,} iterations of each method...\n")

    time_fields = timeit.timeit(method_field_by_field, number=ITERATIONS)
    print(f"-> Field-by-Field Access : {time_fields:.4f} seconds")

    time_bulk = timeit.timeit(method_bulk_unpack, number=ITERATIONS)
    print(f"-> Bulk Struct Unpack    : {time_bulk:.4f} seconds")

    if time_bulk < time_fields:
        speedup = time_fields / time_bulk
        print(f"\n[+] Bulk Unpack is {speedup:.2f}x faster!")
    else:
        print("\n[!] Unexpected performance regression.")

if __name__ == "__main__":
    run_profiler()