import gc
import json
import time
import tracemalloc

try:
    import orjson
except ImportError:
    orjson = None

import msgspec

# Import the new ISFData struct from your refactored codebase
from dwarffi.types import ISFData

def generate_heavy_isf_payload() -> bytes:
    print("Generating synthetic heavy ISF payload...")
    isf = {
        "metadata": {"format": "1.0.0", "producer": {"name": "benchmark"}},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
            "pointer": {"kind": "pointer", "size": 8, "endian": "little"}
        },
        "user_types": {},
        "enums": {},
        "symbols": {},
        "typedefs": {}
    }

    # Generate 15,000 Structs
    for i in range(15000):
        isf["user_types"][f"struct_heavy_{i}"] = {
            "kind": "struct",
            "size": 16,
            "fields": {
                "field_a": {"offset": 0, "type": {"kind": "base", "name": "int"}},
                "field_b": {"offset": 8, "type": {"kind": "pointer", "subtype": {"kind": "base", "name": "int"}}}
            }
        }
        
    # Generate 5,000 Enums
    for i in range(5000):
        isf["enums"][f"enum_heavy_{i}"] = {
            "size": 4,
            "base": "int",
            "constants": {"A": 0, "B": 1, "C": 2}
        }

    # Generate 50,000 Symbols
    for i in range(50000):
        isf["symbols"][f"sys_func_{i}"] = {
            "address": 0xFFFFFFFF81000000 + (i * 16),
            "type": {"kind": "struct", "name": f"struct_heavy_{i % 15000}"}
        }

    payload = json.dumps(isf).encode("utf-8")
    print(f"Payload generated: {len(payload) / 1024 / 1024:.2f} MB\n")
    return payload


def parse_old_way_json(data: bytes):
    """Simulates the legacy VtypeJson parsing and validation using standard json."""
    raw_data = json.loads(data)
    
    # Legacy manual schema validation
    required_sections = ["base_types", "user_types"]
    missing = [s for s in required_sections if s not in raw_data]
    if missing:
        raise ValueError(f"ISF is missing required top-level sections: {missing}")

    for name, definition in raw_data.get("user_types", {}).items():
        if "kind" not in definition:
            raise ValueError(f"User type '{name}' is missing the required 'kind' field.")
            
    return raw_data


def parse_old_way_orjson(data: bytes):
    """Simulates the legacy VtypeJson parsing and validation using orjson (if available)."""
    raw_data = orjson.loads(data)
    
    required_sections = ["base_types", "user_types"]
    missing = [s for s in required_sections if s not in raw_data]
    if missing:
        raise ValueError(f"ISF is missing required top-level sections: {missing}")

    for name, definition in raw_data.get("user_types", {}).items():
        if "kind" not in definition:
            raise ValueError(f"User type '{name}' is missing the required 'kind' field.")
            
    return raw_data


def parse_new_way_msgspec(data: bytes):
    """Simulates the new msgspec strict-schema decoding."""
    return msgspec.json.decode(data, type=ISFData)


def run_benchmark(name: str, func, payload: bytes, iterations: int = 5):
    print(f"--- Benchmarking: {name} ---")
    
    # 1. Measure Time
    gc.disable() # Disable GC to isolate parsing CPU time
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = func(payload)
    end_time = time.perf_counter()
    gc.enable()
    
    avg_time = (end_time - start_time) / iterations
    
    # 2. Measure Memory
    gc.collect()
    tracemalloc.start()
    
    parsed_obj = func(payload)
    
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    # Keep object alive intentionally so 'current_mem' reflects retained size
    _ = parsed_obj 
    
    print(f"  Average Time : {avg_time:.4f} seconds")
    print(f"  Peak Memory  : {peak_mem / 1024 / 1024:.2f} MB")
    print(f"  Retained Mem : {current_mem / 1024 / 1024:.2f} MB\n")
    
    return avg_time, peak_mem, current_mem


def main():
    payload = generate_heavy_isf_payload()
    iterations = 10
    
    time_json, peak_json, ret_json = run_benchmark("Legacy Parser (Standard 'json')", parse_old_way_json, payload, iterations)
    
    if orjson:
        time_orjson, peak_orjson, ret_orjson = run_benchmark("Legacy Parser ('orjson')", parse_old_way_orjson, payload, iterations)
    else:
        print("--- Benchmarking: Legacy Parser ('orjson') ---\n  [orjson not installed, skipping]\n")
        time_orjson = float('inf')

    time_msgspec, peak_msgspec, ret_msgspec = run_benchmark("New Parser ('msgspec')", parse_new_way_msgspec, payload, iterations)

    print("================ SUMMARY ================")
    print(f"Speedup vs standard json : {time_json / time_msgspec:.2f}x faster")
    if orjson:
        print(f"Speedup vs orjson        : {time_orjson / time_msgspec:.2f}x faster")
    
    print(f"Peak Memory Reduction    : {peak_json / peak_msgspec:.2f}x less RAM used during parsing")
    print(f"Retained Memory Reduction: {ret_json / ret_msgspec:.2f}x less RAM held by parsed objects")
    print("=========================================")

if __name__ == "__main__":
    main()