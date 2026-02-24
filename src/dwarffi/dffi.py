import json
import lzma
import os
import re
import shutil
import subprocess
import tempfile
import weakref
from typing import Any, Dict, List, Optional, Union

from dwarffi.core import (
    BoundArrayView,
    BoundTypeInstance,
    Ptr,
    VtypeBaseType,
    VtypeEnum,
    VtypeUserType,
    isf_from_dict,
    load_isf_json,
)


class DFFI:
    def __init__(self, isf_path: Optional[str] = None):
        self._file_order = []
        self.vtypejsons = {}
        if isf_path:
            self.load_isf(isf_path)

    def load_isf(self, path: str):
        if path not in self.vtypejsons:
            self._file_order.append(path)
            self.vtypejsons[path] = load_isf_json(path)
    
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

    def get_symbol(self, name: str):
        for f in self._file_order:
            res = self.vtypejsons[f].get_symbol(name)
            if res and not (hasattr(res, "address") and res.address in [None, 0]):
                return res

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

    def _create_instance(self, type_input, buffer, instance_offset_in_buffer=0):
        for f in self._file_order:
            try:
                return self.vtypejsons[f].create_instance(
                    type_input, buffer, instance_offset_in_buffer
                )
            except ValueError:
                continue
        raise ValueError("Type definition not found in any loaded ISF.")

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

    def sizeof(self, ctype: Union[str, BoundTypeInstance, Any]) -> int:
        t = self.typeof(ctype) if isinstance(ctype, str) else ctype

        if isinstance(t, BoundTypeInstance):
            size = t._instance_type_def.size
        elif isinstance(t, dict):
            if t.get("kind") == "pointer":
                size = self.get_base_type("pointer").size
            elif t.get("kind") == "array":
                subtype_name = t.get("subtype", {}).get("name")
                if not subtype_name:
                    raise ValueError(f"Array subtype missing name in {t}")
                elem_size = self.sizeof(subtype_name)
                size = elem_size * t.get("count", 0)
            else:
                size = self.get_type_size(t)
        elif hasattr(t, "size"):
            size = t.size
        else:
            raise TypeError(f"Cannot determine size of {ctype}")

        if size is None:
            raise ValueError(f"Type '{ctype}' has an unknown or undefined size.")
        return size
 
    def offset(self, cdata: BoundTypeInstance) -> int:
        """Returns the offset of the given cdata instance within its underlying buffer."""
        return cdata._instance_offset

    def offsetof(self, ctype: str, *fields_or_indexes) -> int:
        t = self.typeof(ctype)
        if not isinstance(t, VtypeUserType):
            raise TypeError(f"Type '{ctype}' is not a struct or union.")

        offset = 0
        current_type = t

        for field_name in fields_or_indexes:
            if isinstance(current_type, VtypeUserType):
                # 4. Search including anonymous fields recursively
                def _find_field_recursive(t_def, name, current_off):
                    if name in t_def.fields:
                        f = t_def.fields[name]
                        return f, current_off + f.offset
                    for f in t_def.fields.values():
                        if f.anonymous:
                            sub_t = self.get_type(f.type_info.get("name"))
                            if isinstance(sub_t, VtypeUserType):
                                res = _find_field_recursive(sub_t, name, current_off + f.offset)
                                if res[0]:
                                    return res
                    return None, None

                field, field_offset = _find_field_recursive(current_type, field_name, 0)
                if not field:
                    raise KeyError(f"Type '{current_type.name}' has no field '{field_name}'")

                offset += field_offset

                # Update current_type for nested structures
                if field.type_info.get("kind") in ["struct", "union"]:
                    current_type = self.get_type(field.type_info.get("name"))
            else:
                raise TypeError(f"Cannot get offset of '{field_name}' inside non-struct type.")

        return offset

    def addressof(self, cdata: BoundTypeInstance, *fields_or_indexes) -> Ptr:
        """
        Returns the offset/address of the buffer. In a rehosting context,
        this represents the local buffer offset unless mapped via a memory backend.
        """
        base_addr = cdata._instance_offset
        subtype_name = cdata._instance_type_name

        if fields_or_indexes:
            base_addr += self.offsetof(cdata._instance_type_name, *fields_or_indexes)
            t = self.typeof(cdata._instance_type_name)
            current_type = t
            for field_name in fields_or_indexes:

                def _find_type(t_def, name):
                    if name in t_def.fields:
                        return t_def.fields[name].type_info
                    for f in t_def.fields.values():
                        if f.anonymous:
                            sub_t = self.get_type(f.type_info.get("name"))
                            if isinstance(sub_t, VtypeUserType):
                                res = _find_type(sub_t, name)
                                if res:
                                    return res
                    return None

                type_info = _find_type(current_type, field_name)
                subtype_name = type_info.get("name") if type_info else "void"
                if type_info and type_info.get("kind") in ["struct", "union"]:
                    current_type = self.get_type(subtype_name)
                else:
                    current_type = None

        return Ptr(base_addr, {"name": subtype_name}, self)

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

    def cast(self, ctype: str, value: Any) -> Union[BoundTypeInstance, Ptr]:
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

        # Re-casting an existing buffer to a new type
        if isinstance(value, BoundTypeInstance):
            return self._create_instance(t, value._instance_buffer, value._instance_offset)

        raise TypeError(f"Cannot cast {type(value)} to {ctype}")

    def from_buffer(
        self,
        ctype: str,
        python_buffer: Union[bytearray, memoryview, bytes],
        require_writable: bool = False,
    ) -> BoundTypeInstance:
        if require_writable and isinstance(python_buffer, bytes):
            raise TypeError("Buffer is read-only")

        t = self.typeof(ctype)
        # Convert bytes to bytearray to satisfy BoundTypeInstance requirement if read-only is tolerated
        if isinstance(python_buffer, bytes):
            python_buffer = bytearray(python_buffer)

        return self._create_instance(t, python_buffer)

    def buffer(self, cdata: BoundTypeInstance, size: Optional[int] = None) -> memoryview:
        """Returns a memoryview over the underlying bytearray of the cdata."""
        size = size if size is not None else cdata._instance_type_def.size
        start = cdata._instance_offset
        return memoryview(cdata._instance_buffer)[start : start + size]
    
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

    def string(
        self, cdata: Union[BoundTypeInstance, Ptr, BoundArrayView], maxlen: int = 4096
    ) -> bytes:
        if isinstance(cdata, BoundArrayView):
            cdata = cdata._parent_instance

        if isinstance(cdata, BoundTypeInstance):
            raw_bytes = cdata._instance_buffer[cdata._instance_offset :]
            if maxlen > 0:
                raw_bytes = raw_bytes[:maxlen]
            null_idx = raw_bytes.find(b"\x00")
            return bytes(raw_bytes[:null_idx] if null_idx != -1 else raw_bytes)

        if isinstance(cdata, Ptr):
            # Without a memory backend plugged in, a raw Ptr can't be dereferenced for string reading.
            raise NotImplementedError(
                "Cannot read string from raw Ptr address without an attached memory backend."
            )

    def unpack(self, cdata: Union[BoundArrayView, BoundTypeInstance], length: int) -> list:
        if isinstance(cdata, BoundArrayView):
            return [cdata[i] for i in range(min(length, len(cdata)))]
        raise TypeError("unpack() currently requires an array view.")

    def gc(
        self, cdata: BoundTypeInstance, destructor: callable, size: int = 0
    ) -> BoundTypeInstance:
        """
        Attach a finalizer to the cdata object. When garbage collected,
        destructor(cdata) will be executed.
        """
        weakref.finalize(cdata, destructor, cdata)
        return cdata

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
            vtype_obj = isf_from_dict(isf_dict)

            pseudo_path = f"<cdef_{id(source)}>"
            self._file_order.append(pseudo_path)
            self.vtypejsons[pseudo_path] = vtype_obj
