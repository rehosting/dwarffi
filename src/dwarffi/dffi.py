import json
import lzma
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Union

from .instances import BoundArrayView, BoundTypeInstance, Ptr
from .parser import VtypeJson
from .types import VtypeBaseType, VtypeEnum, VtypeSymbol, VtypeUserType


class DFFI:
    def __init__(self, isf_input: Optional[Union[str, dict, list]] = None):
        self._file_order = []
        self.vtypejsons = {}

        if isf_input is not None:
            if isinstance(isf_input, list):
                for item in isf_input:
                    self.load_isf(item)
            else:
                self.load_isf(isf_input)

    def load_isf(self, isf_input: Union[str, dict]):
        """
        Loads a singular ISF definition from a file path or a direct dictionary.
        """
        if isinstance(isf_input, dict):
            # Generate a unique pseudo-path for the dictionary entry
            pseudo_path = f"<dict_{id(isf_input)}>"
            if pseudo_path not in self.vtypejsons:
                self._file_order.append(pseudo_path)
                self.vtypejsons[pseudo_path] = VtypeJson(isf_input)
        elif isinstance(isf_input, str):
            if isf_input not in self.vtypejsons:
                self._file_order.append(isf_input)
                self.vtypejsons[isf_input] = VtypeJson(isf_input)
        else:
            raise TypeError("load_isf expects a file path (str) or a dictionary (dict)")
    
    def _resolve_type_info(self, type_info: Dict[str, Any]) -> Dict[str, Any]:
        visited = set()
        current = type_info
        while current and current.get("kind") == "typedef":
            name = current.get("name")
            if not name:
                break
            if name in visited:
                raise ValueError(f"Circular typedef: {name}")
            visited.add(name)

            td = None
            for f in self._file_order:
                td = self.vtypejsons[f]._raw_typedefs.get(name)
                if td:
                    break
            if not td:
                break
            current = td
        return current

    def get_base_type(self, name: str):
        for f in self._file_order:
            if res := self.vtypejsons[f].get_base_type(name):
                return res

    def get_user_type(self, name: str):
        for f in self._file_order:
            if res := self.vtypejsons[f].get_user_type(name):
                return res

    def get_enum(self, name: str):
        for f in self._file_order:
            if res := self.vtypejsons[f].get_enum(name):
                return res

    def get_symbol(self, name: str, strict: bool = False) -> Optional[VtypeSymbol]:
        """
        Look up a symbol by name across all loaded ISFs.
        """
        for f in self._file_order:
            if sym := self.vtypejsons[f].get_symbol(name):
                return sym
        
        if strict:
            raise KeyError(f"Symbol '{name}' not found in any loaded ISF files ({self._file_order}).")
        return None

    def get_type(self, name: str):
        for f in self._file_order:
            if res := self.vtypejsons[f].get_type(name):
                return res

    def get_symbols_by_address(self, target_address: int):
        results = []
        for f in self._file_order:
            results.extend(self.vtypejsons[f].get_symbols_by_address(target_address))
        return results

    def get_type_size(self, in_type_info: dict):
        type_info = self._resolve_type_info(in_type_info)
        for f in self._file_order:
            if res := self.vtypejsons[f].get_type_size(type_info):
                return res
    
    def _create_instance(
        self,
        type_input: Union[str, VtypeUserType, VtypeBaseType, VtypeEnum],
        buffer: Union[bytes, bytearray, memoryview],
        instance_offset_in_buffer: int = 0,
    ) -> BoundTypeInstance:
        """
        Creates a BoundTypeInstance by resolving the type across all loaded ISFs
        and binding it to the provided buffer.
        """
        if isinstance(buffer, bytes):
            processed_buffer = bytearray(buffer)
        elif isinstance(buffer, (bytearray, memoryview)):
            processed_buffer = buffer
        else:
            raise TypeError("Input buffer must be bytes, bytearray, or memoryview.")

        if isinstance(type_input, str):
            type_name = type_input
            type_def = self.get_type(type_input)
        else:
            type_def = type_input
            type_name = getattr(type_def, "name", "unknown")

        if type_def is None:
            raise ValueError(f"Type definition for '{type_name}' not found in any loaded ISF.")

        # Validate size
        if not hasattr(type_def, "size") or type_def.size is None:
            type_kind = getattr(type_def, "kind", None)
            if not (type_kind == "void" and getattr(type_def, "size", None) == 0):
                raise ValueError(f"Type definition for '{type_name}' lacks a valid size.")

        # Bounds checking
        if type_def.size is not None:
            effective_len = len(processed_buffer) - instance_offset_in_buffer
            if type_def.size > effective_len:
                raise ValueError(
                    f"Buffer too small for '{type_name}' at offset {instance_offset_in_buffer}. "
                    f"Needs {type_def.size} bytes, got {effective_len}."
                )

        # Instantiate, passing 'self' (the DFFI object) as the type accessor
        return BoundTypeInstance(
            type_name, type_def, processed_buffer, self, instance_offset_in_buffer
        )

    def shift_symbol_addresses(self, delta: int, path: str = None):
        if path is None:
            for f in self._file_order:
                self.vtypejsons[f].shift_symbol_addresses(delta)
        else:
            self.vtypejsons[path].shift_symbol_addresses(delta)

    def _make_subtype_info(self, base_name: str) -> dict:
        """Helper to create ISF-compatible type_info references."""
        base_t = self.typeof(base_name)
        if isinstance(base_t, dict):
            return base_t
        elif isinstance(base_t, VtypeBaseType):
            return {"kind": "base", "name": base_t.name}
        elif isinstance(base_t, VtypeUserType):
            return {"kind": base_t.kind, "name": base_t.name}
        elif isinstance(base_t, VtypeEnum):
            return {"kind": "enum", "name": base_t.name}
        return {"name": base_name}

    def typeof(
        self, ctype: Union[str, BoundTypeInstance, Ptr, BoundArrayView]
    ) -> Union[VtypeUserType, VtypeBaseType, VtypeEnum, Dict]:
        if isinstance(ctype, BoundTypeInstance):
            return ctype._instance_type_def
        if isinstance(ctype, Ptr):
            return {"kind": "pointer", "subtype": ctype.points_to_type_info}
        if isinstance(ctype, BoundArrayView):
            return {
                "kind": "array",
                "count": ctype._array_count,
                "subtype": ctype._array_subtype_info,
            }

        if isinstance(ctype, str):
            ctype = ctype.strip()

            # 1. Strip C-style keywords ("struct ", "union ", "enum ") for the lookup
            # but keep a copy for the regex parsers
            lookup_name = ctype
            for prefix in ["struct ", "union ", "enum "]:
                if lookup_name.startswith(prefix):
                    lookup_name = lookup_name[len(prefix) :].strip()
                    break

            # 2. Dynamic Array parsing
            m = re.match(r"^(.*?)\[(\d*)\]$", ctype)
            if m:
                base = m.group(1).strip()
                count = int(m.group(2)) if m.group(2) else 0
                subtype_info = self._make_subtype_info(base)
                return {"kind": "array", "count": count, "subtype": subtype_info}

            # Pointer parsing
            if ctype.endswith("*"):
                base_name = ctype[:-1].strip()
                subtype_info = self._make_subtype_info(base_name)
                return {"kind": "pointer", "subtype": subtype_info}

            # 3. Resolve Typedefs / Raw Types
            # Use the stripped 'lookup_name' for the ISF search
            resolved_info = self._resolve_type_info({"kind": "typedef", "name": lookup_name})

            if resolved_info.get("kind") == "typedef":
                t = self.get_type(lookup_name)
                if not t:
                    raise KeyError(f"Unknown DWARF type: '{ctype}'")
                return t
            elif resolved_info.get("kind") in ("pointer", "array"):
                return resolved_info
            else:
                t = self.get_type(resolved_info["name"])
                if not t:
                    raise KeyError(
                        f"Resolved typedef '{ctype}' to unknown target '{resolved_info['name']}'"
                    )
                return t

        raise TypeError(
            f"Expected string, BoundTypeInstance, Ptr, or BoundArrayView, got {type(ctype)}"
        )

    def sizeof(self, ctype: Union[str, BoundTypeInstance, Ptr, BoundArrayView, Any]) -> int:
        """
        Returns the size in bytes of the given type or instance.
        """
        if isinstance(ctype, (str, Ptr, BoundArrayView)):
            t = self.typeof(ctype)
        else:
            t = ctype

        size = None
        if isinstance(t, BoundTypeInstance):
            size = t._instance_type_def.size
        elif isinstance(t, dict):
            kind = t.get("kind")
            if kind == "pointer":
                ptr_def = self.get_base_type("pointer")
                if ptr_def is None:
                    raise KeyError("Cannot determine pointer size: base type 'pointer' not found in loaded ISF files.")
                size = ptr_def.size if ptr_def else None
            elif kind == "array":
                subtype = t.get("subtype")
                if not subtype:
                    raise ValueError(f"Array type missing subtype information: {t}")
                elem_size = self.sizeof(subtype)
                size = elem_size * t.get("count", 0)
            else:
                size = self.get_type_size(t)
        elif hasattr(t, "size"):
            size = t.size
        else:
            raise TypeError(f"Cannot determine size of {type(ctype).__name__}: {ctype}")

        if size is None:
            raise ValueError(f"Type '{ctype}' has an unknown or undefined size.")
        return size
 
    def offset(self, cdata: BoundTypeInstance) -> int:
        """Returns the offset of the given cdata instance within its underlying buffer."""
        return cdata._instance_offset

    def offsetof(self, ctype: str, *fields_or_indexes: str) -> int:
        """
        Returns the offset of a field (or nested field path) within a struct or union.
        Supports recursive searching of anonymous members.
        """
        t = self.typeof(ctype)
        if not isinstance(t, VtypeUserType):
            raise TypeError(f"offsetof() requires a struct or union type, got {type(t).__name__} ('{ctype}')")

        current_offset = 0
        current_type = t

        for field_name in fields_or_indexes:
            if not isinstance(current_type, VtypeUserType):
                raise TypeError(f"Cannot get offset of '{field_name}' inside non-struct type '{getattr(current_type, 'name', 'unknown')}'")

            # Recursive helper to find field through anonymous members
            def _find_field_recursive(t_def, name, accum_off):
                # 1. Direct hit
                if name in t_def.fields:
                    f = t_def.fields[name]
                    return f, accum_off + f.offset
                
                # 2. Search anonymous children
                for f in t_def.fields.values():
                    if f.anonymous:
                        # Resolve the anonymous member's type
                        sub_t_info = self._resolve_type_info(f.type_info)
                        sub_t = self.get_type(sub_t_info.get("name"))
                        
                        if isinstance(sub_t, VtypeUserType):
                            res_field, res_off = _find_field_recursive(sub_t, name, accum_off + f.offset)
                            if res_field:
                                return res_field, res_off
                return None, None

            field, field_offset = _find_field_recursive(current_type, field_name, 0)
            
            if not field:
                raise AttributeError(f"Type '{current_type.name}' has no field named '{field_name}' (checked anonymous fields recursively)")

            current_offset += field_offset
            
            # Update current_type for the next segment in the path
            next_t_info = self._resolve_type_info(field.type_info)
            if next_t_info.get("kind") in ["struct", "union"]:
                current_type = self.get_type(next_t_info.get("name"))
            else:
                current_type = next_t_info

        return current_offset

    def addressof(self, cdata: BoundTypeInstance, *fields_or_indexes: str) -> Ptr:
        """
        Returns a Ptr to the given cdata or a nested field within it.
        """
        base_addr = cdata._instance_offset
        # Default to pointing to the instance's own type
        target_type_info = cdata._instance_type_def

        if fields_or_indexes:
            # 1. Calculate the absolute buffer offset using recursive offsetof
            base_addr += self.offsetof(cdata._instance_type_name, *fields_or_indexes)
            
            # 2. Resolve the final type info in the path for the Ptr object
            current_type = cdata._instance_type_def
            for field_name in fields_or_indexes:
                
                def _find_type_recursive(t_def, name):
                    if name in t_def.fields:
                        return t_def.fields[name].type_info
                    for f in t_def.fields.values():
                        if f.anonymous:
                            sub_t_info = self._resolve_type_info(f.type_info)
                            sub_t = self.get_type(sub_t_info.get("name"))
                            if isinstance(sub_t, VtypeUserType):
                                res = _find_type_recursive(sub_t, name)
                                if res:
                                    return res
                    return None

                if not isinstance(current_type, VtypeUserType):
                    break # Should be caught by offsetof, but safety first
                
                field_type_info = _find_type_recursive(current_type, field_name)
                target_type_info = self._resolve_type_info(field_type_info)
                
                # If nested, continue the search in the next struct
                if target_type_info.get("kind") in ["struct", "union"]:
                    current_type = self.get_type(target_type_info.get("name"))
                else:
                    current_type = target_type_info

        return Ptr(base_addr, target_type_info, self)

    def _deep_init(self, instance: Any, init: Any):
        """3. Deep Struct Initialization."""
        if isinstance(init, dict) and isinstance(instance, BoundTypeInstance):
            for k, v in init.items():
                if hasattr(instance, k):
                    field_val = getattr(instance, k)
                    if isinstance(field_val, (BoundTypeInstance, BoundArrayView)) and isinstance(
                        v, (dict, list)
                    ):
                        self._deep_init(field_val, v)
                    else:
                        setattr(instance, k, v)
        elif isinstance(init, list) and isinstance(instance, BoundArrayView):
            for i, v in enumerate(init):
                if isinstance(instance[i], (BoundTypeInstance, BoundArrayView)) and isinstance(
                    v, (dict, list)
                ):
                    self._deep_init(instance[i], v)
                else:
                    instance[i] = v
        elif isinstance(instance, BoundTypeInstance):
            instance[0] = init

    def new(self, ctype: str, init: Any = None) -> Union[BoundTypeInstance, BoundArrayView]:
        t = self.typeof(ctype)

        # 2. Handle dynamic arrays natively
        if isinstance(t, dict) and t.get("kind") == "array":
            if init is not None:
                if isinstance(init, (bytes, bytearray, str)):
                    if isinstance(init, str):
                        init = init.encode("utf-8")
                    if t.get("count") == 0:
                        t["count"] = len(init) + 1  # Add null terminator for C-strings
                elif isinstance(init, list):
                    if t.get("count") == 0:
                        t["count"] = len(init)

            size = self.sizeof(t)
            buf = bytearray(size)

            # Create a dummy struct to hold the array so BoundArrayView works flawlessly
            # without modifying the core instances engine.
            dummy_name = f"__dummy_{id(buf)}"
            primary_isf_path = self._file_order[0]
            self.vtypejsons[primary_isf_path]._raw_user_types[dummy_name] = {
                "kind": "struct",
                "size": size,
                "fields": {"arr": {"offset": 0, "type": t}},
            }
            self.vtypejsons[primary_isf_path]._parsed_user_types_cache.pop(dummy_name, None)

            instance = self._create_instance(dummy_name, buf)
            arr_view = instance.arr

            if init is not None:
                if isinstance(init, (bytes, bytearray)):
                    buf[: len(init)] = init
                elif isinstance(init, list):
                    self._deep_init(arr_view, init)

            return arr_view

        if getattr(t, "size", None) is None:
            raise ValueError(f"Cannot allocate memory for type '{ctype}' with unknown size.")

        buf = bytearray(t.size)
        instance = self._create_instance(t, buf)

        if init is not None:
            self._deep_init(instance, init)

        return instance

    def cast(self, ctype: str, value: Any) -> Union[BoundTypeInstance, Ptr, BoundArrayView]:
        """
        Interpret 'value' as the specified C type.
        """
        t = self.typeof(ctype)

        # Casting an integer to a pointer
        if isinstance(value, int):
            if isinstance(t, dict) and t.get("kind") == "pointer":
                return Ptr(value, t.get("subtype"), self)

            if hasattr(t, "size"):
                buf = bytearray(t.size)
            else:
                buf = bytearray(8)
            instance = self._create_instance(t, buf)
            instance[0] = value
            return instance

        # Re-casting an existing memory buffer to a new type seamlessly
        if isinstance(value, BoundTypeInstance):
            return self.from_buffer(ctype, value._instance_buffer, offset=value._instance_offset)
        elif isinstance(value, BoundArrayView):
            return self.from_buffer(
                ctype, 
                value._parent_instance._instance_buffer, 
                offset=value._parent_instance._instance_offset + value._array_start_offset_in_parent
            )

        raise TypeError(f"Cannot cast {type(value)} to {ctype}")

    def from_buffer(
        self,
        ctype: str,
        python_buffer: Union[bytearray, memoryview, bytes],
        offset: int = 0,
        require_writable: bool = False,
    ) -> Union[BoundTypeInstance, BoundArrayView]:
        if require_writable and isinstance(python_buffer, bytes):
            raise TypeError("Buffer is read-only")

        t = self.typeof(ctype)
        if isinstance(python_buffer, bytes):
            python_buffer = bytearray(python_buffer)

        # Handle pointers and dynamic arrays natively by wrapping them in an array view
        if isinstance(t, dict) and t.get("kind") in ("array", "pointer"):
            if t.get("kind") == "pointer":
                # Treat a bound pointer like an unbounded array of that pointer type
                t = {"kind": "array", "count": 0, "subtype": t}

            elem_size = self.sizeof(t.get("subtype"))
            if elem_size == 0:
                elem_size = 1

            count = t.get("count", 0)
            if count == 0:
                count = (len(python_buffer) - offset) // elem_size
                t["count"] = count

            dummy_size = count * elem_size
            
            # Use hash(str(t)) to prevent cache collisions if the same buffer is cast to multiple types
            dummy_name = f"__dummy_{id(python_buffer)}_{offset}_{hash(str(t))}"
            primary_isf_path = self._file_order[0]
            
            self.vtypejsons[primary_isf_path]._raw_user_types[dummy_name] = {
                "kind": "struct",
                "size": dummy_size,
                "fields": {"arr": {"offset": 0, "type": t}},
            }
            self.vtypejsons[primary_isf_path]._parsed_user_types_cache.pop(dummy_name, None)

            instance = self._create_instance(dummy_name, python_buffer, instance_offset_in_buffer=offset)
            return instance.arr

        return self._create_instance(t, python_buffer, instance_offset_in_buffer=offset)

    def buffer(self, cdata: Union[BoundTypeInstance, Ptr, BoundArrayView], size: int = -1) -> memoryview:
        """
        Returns a zero-copy memoryview of the underlying buffer.
        """
        if isinstance(cdata, Ptr):
            # A pointer doesn't own memory, it just points. 
            # In a real rehosting environment, this would call out to QEMU's memory read API.
            raise TypeError("Cannot get a direct buffer from a Ptr. Dereference it first if bound to local memory.")
            
        buf = cdata._instance_buffer if hasattr(cdata, "_instance_buffer") else cdata._parent_instance._instance_buffer
        offset = cdata._instance_offset if hasattr(cdata, "_instance_offset") else cdata._parent_instance._instance_offset + cdata._array_start_offset_in_parent
        
        if size == -1:
            size = self.sizeof(cdata)
            
        # Return a zero-copy slice
        return memoryview(buf)[offset : offset + size]

    def to_bytes(self, cdata: BoundTypeInstance) -> bytes:
        """Returns the underlying bytes of the given cdata instance."""
        size = cdata._instance_type_def.size
        if size == 0:
            return b""
        return bytes(cdata)

    def memmove(
        self,
        dest: Union[BoundTypeInstance, bytearray],
        src: Union[BoundTypeInstance, bytearray, bytes],
        n: int,
    ):
        """Copy n bytes from src to dest."""
        dest_buf = dest._instance_buffer if isinstance(dest, BoundTypeInstance) else dest
        dest_off = dest._instance_offset if isinstance(dest, BoundTypeInstance) else 0

        src_buf = src._instance_buffer if isinstance(src, BoundTypeInstance) else src
        src_off = src._instance_offset if isinstance(src, BoundTypeInstance) else 0

        dest_buf[dest_off : dest_off + n] = src_buf[src_off : src_off + n]

    def string(self, cdata: Union[BoundTypeInstance, Ptr, BoundArrayView], maxlen: int = -1) -> bytes:
        """
        Reads a null-terminated string from memory, or exactly maxlen bytes.
        """
        # Get the fast memoryview
        mem = self.buffer(cdata, maxlen if maxlen > 0 else len(cdata._instance_buffer) - cdata._instance_offset)
        
        # If maxlen is specified, just return the raw bytes
        if maxlen > 0:
            return mem.tobytes()
            
        # Otherwise, search for the null terminator rapidly in C
        byte_data = mem.tobytes()
        null_idx = byte_data.find(b'\x00')
        
        if null_idx == -1:
            return byte_data # No null terminator found, return everything we have
        return byte_data[:null_idx]

    def unpack(self, cdata: Union[BoundArrayView, BoundTypeInstance], length: int) -> list:
        if isinstance(cdata, BoundArrayView):
            return [cdata[i] for i in range(min(length, len(cdata)))]
        raise TypeError("unpack() currently requires an array view.")

    def cdef(
        self,
        source: str,
        compiler: str = "gcc",
        compiler_flags: Optional[List[str]] = None,
        dwarf2json_cmd: str = "dwarf2json",
        save_isf_to: Optional[str] = None,
    ):
        """
        Compile C code on the fly, extract DWARF info via dwarf2json,
        and load the resulting types into this DFFI instance.

        :param source: The raw C code string to compile.
        :param compiler: The compiler executable (e.g., 'gcc', 'clang', 'arm-none-eabi-gcc').
        :param compiler_flags: List of flags to pass to the compiler. Defaults to ['-O0', '-g', '-gdwarf-4', '-fno-eliminate-unused-debug-types', '-c'].
        :param dwarf2json_cmd: The name or path of the dwarf2json executable.
        :param save_isf_to: Optional file path to save the generated ISF (supports .json and .json.xz).
        """
        if not shutil.which(dwarf2json_cmd):
            raise RuntimeError(
                f"'{dwarf2json_cmd}' not found in PATH.\n"
                "dwarffi requires dwarf2json to extract type info from compiled C code.\n"
                "Please download or build it from: https://github.com/volatilityfoundation/dwarf2json"
            )

        compiler_exe = compiler.split()[0]
        if not shutil.which(compiler_exe):
            raise RuntimeError(f"Compiler '{compiler_exe}' not found in PATH.")

        if compiler_flags is None:
            compiler_flags = ["-O0", "-g", "-gdwarf-4", "-fno-eliminate-unused-debug-types", "-c"]

        with tempfile.TemporaryDirectory() as tmpdir:
            c_file = os.path.join(tmpdir, "source.c")
            # You are compiling with -c, so this should be an object file
            o_file = os.path.join(tmpdir, "source.o")

            with open(c_file, "w", encoding="utf-8") as f:
                f.write(source)

            # 1) Compile (object file)
            cmd_compile = compiler.split() + compiler_flags + [c_file, "-o", o_file]
            try:
                subprocess.run(cmd_compile, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    f"Compilation failed:\nCommand: {' '.join(cmd_compile)}\nStderr: {e.stderr}"
                ) from e

            # 2) Run dwarf2json (types first; fallback to --elf if needed)
            cmd_d2j = [dwarf2json_cmd, "linux", "--elf-types", o_file]
            try:
                res = subprocess.run(cmd_d2j, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError:
                cmd_d2j = [dwarf2json_cmd, "linux", "--elf", o_file]
                try:
                    res = subprocess.run(cmd_d2j, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        f"dwarf2json failed:\nCommand: {' '.join(cmd_d2j)}\nStderr: {e.stderr}"
                    ) from e

            # 3) Parse
            try:
                isf_dict = json.loads(res.stdout)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Failed to parse dwarf2json output: {e}\nOutput head: {res.stdout[:500]}"
                ) from e

            # 4) Optionally save the ISF to disk
            if save_isf_to:
                out_path = os.path.abspath(save_isf_to)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

                if out_path.endswith(".json.xz"):
                    with lzma.open(out_path, "wt", encoding="utf-8") as xf:
                        json.dump(isf_dict, xf, indent=2, sort_keys=True)
                elif out_path.endswith(".json"):
                    with open(out_path, "w", encoding="utf-8") as jf:
                        json.dump(isf_dict, jf, indent=2, sort_keys=True)
                else:
                    raise ValueError("save_isf_to must end with '.json' or '.json.xz'")

            # 5) Load into this DFFI instance
            vtype_obj = VtypeJson(isf_dict)

            pseudo_path = f"<cdef_{id(source)}>"
            self._file_order.append(pseudo_path)
            self.vtypejsons[pseudo_path] = vtype_obj
