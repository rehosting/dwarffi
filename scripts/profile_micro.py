import timeit

from dwarffi.dffi import DFFI


def run_memory_profiler():
    print("[*] Setting up Memory/Array benchmark...")
    
    isf = {
        "metadata": {},
        "base_types": {
            "u8": {"kind": "int", "size": 1, "signed": False, "endian": "little"},
            "u32": {"kind": "int", "size": 4, "signed": False, "endian": "little"},
        },
        "user_types": {}, "enums": {}, "symbols": {}
    }
    
    ffi = DFFI(isf)
    
    # Simulate a 4KB memory page of integers (1000 u32s)
    ARRAY_SIZE = 1000
    arr = ffi.new(f"u32[{ARRAY_SIZE}]")
    
    for i in range(ARRAY_SIZE):
        arr[i] = i

    def method_python_iteration():
        # The naive way: hitting __getitem__ 1000 times
        return [arr[i] for i in range(ARRAY_SIZE)]

    def method_ffi_unpack():
        # The fast C-level struct unpack we just built
        return ffi.unpack(arr)

    def method_zero_copy_buffer():
        # Directly grabbing the underlying memoryview (Step 4)
        return bytes(ffi.buffer(arr))

    ITERATIONS = 10_000
    print(f"[*] Running {ITERATIONS:,} iterations...\n")

    t_iter = timeit.timeit(method_python_iteration, number=ITERATIONS)
    print(f"-> Python For-Loop (__getitem__) : {t_iter:.4f} seconds")

    t_unpack = timeit.timeit(method_ffi_unpack, number=ITERATIONS)
    print(f"-> ffi.unpack() (C-level)        : {t_unpack:.4f} seconds")

    t_buffer = timeit.timeit(method_zero_copy_buffer, number=ITERATIONS)
    print(f"-> ffi.buffer() (Zero-copy)      : {t_buffer:.4f} seconds")

if __name__ == "__main__":
    run_memory_profiler()