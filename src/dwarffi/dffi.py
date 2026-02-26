import fnmatch
import json
import lzma
import os
import re
import shutil
import struct
import subprocess
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Union, TypeAlias

from .backend import BytesBackend, LiveMemoryProxy, MemoryBackend
from .instances import BoundArrayView, BoundTypeInstance, Ptr
from .parser import VtypeJson
from .types import VtypeBaseType, VtypeEnum, VtypeSymbol, VtypeUserType

# Clean, unified Type Aliases
VTYPE_CLASSES = (VtypeBaseType, VtypeEnum, VtypeUserType)
BOUND_TYPE_CLASSES = (BoundTypeInstance, BoundArrayView, Ptr)

Vtype: TypeAlias = Union[VtypeBaseType, VtypeEnum, VtypeUserType]
BoundType: TypeAlias = Union[BoundTypeInstance, BoundArrayView, Ptr]


class DFFI:
    """
    DWARFFI (DFFI) Engine.

    Provides a high-performance, CFFI-like interface for interacting with
    binary data and memory using DWARF-derived Intermediate Structure Format (ISF) files.
    """

    def __init__(
        self, 
        isf_input: Optional[Union[str, dict, List[Union[str, dict]]]] = None,
        backend: Optional[Union[MemoryBackend, bytes, bytearray]] = None
    ):
        """
        Initializes the DFFI engine.

        Args:
            isf_input: A file path (str), a dictionary (dict), or a list of paths/dicts
                       containing parsed ISF data. Evaluated in order of provision.
        """
        self._file_order: List[str] = []
        self.vtypejsons: Dict[str, VtypeJson] = {}

        # Configure the backend natively
        if isinstance(backend, (bytes, bytearray)):
            self.backend = BytesBackend(backend)
        else:
            self.backend = backend
        
        # Safely bound LRU cache tied to the instance lifecycle to prevent memory leaks
        self._parse_ctype_string = lru_cache(maxsize=2048)(self._parse_ctype_string_impl)

        if isf_input is not None:
            if isinstance(isf_input, list):
                for item in isf_input:
                    self.load_isf(item)
            else:
                self.load_isf(isf_input)
    
    def _add_vtypejson(self, source: str, vtype_obj: VtypeJson) -> None:
        """Internal helper to add a VtypeJson instance to the engine."""
        self._file_order.append(source)
        self.vtypejsons[source] = vtype_obj
        self._parse_ctype_string.cache_clear()  # Clear the cache to ensure new types are recognized

    def load_isf(self, isf_input: Union[str, dict]) -> None:
        """
        Loads a singular ISF definition from a file path or a direct dictionary.

        Args:
            isf_input: A string path to an ISF file (.json or .json.xz) or a loaded ISF dictionary.
        """
        if isinstance(isf_input, dict):
            # Generate a unique pseudo-path for the dictionary entry
            pseudo_path = f"<dict_{id(isf_input)}>"
            if pseudo_path not in self.vtypejsons:
                self._add_vtypejson(pseudo_path, VtypeJson(isf_input))
        elif isinstance(isf_input, str):
            if isf_input not in self.vtypejsons:
                self._add_vtypejson(isf_input, VtypeJson(isf_input))
        else:
            raise TypeError("load_isf expects a file path (str) or a dictionary (dict)")
    
    @property
    def symbols(self) -> Dict[str, Any]:
        """Returns a dictionary of all symbols across all loaded ISF files."""
        merged = {}
        # Iterate reversed so earlier loaded ISFs correctly overwrite later ones (first-wins resolution)
        for path in reversed(self._file_order):
            for sym_name in self.vtypejsons[path]._raw_symbols.keys():
                sym = self.get_symbol(sym_name)
                if sym:
                    merged[sym_name] = sym
        return merged

    @property
    def types(self) -> Dict[str, "VtypeUserType"]:
        """Returns a dictionary of all user types (structs/unions) across all loaded ISF files."""
        merged = {}
        for path in reversed(self._file_order):
            for type_name in self.vtypejsons[path]._raw_user_types.keys():
                t = self.get_user_type(type_name)
                if t:
                    merged[type_name] = t
        return merged

    @property
    def base_types(self) -> Dict[str, "VtypeBaseType"]:
        """Returns a dictionary of all base types across all loaded ISF files."""
        merged = {}
        for path in reversed(self._file_order):
            for type_name in self.vtypejsons[path]._raw_base_types.keys():
                t = self.get_base_type(type_name)
                if t:
                    merged[type_name] = t
        return merged

    @property
    def enums(self) -> Dict[str, "VtypeEnum"]:
        """Returns a dictionary of all enums across all loaded ISF files."""
        merged = {}
        for path in reversed(self._file_order):
            for enum_name in self.vtypejsons[path]._raw_enums.keys():
                t = self.get_enum(enum_name)
                if t:
                    merged[enum_name] = t
        return merged
    
    def _resolve_type_info(self, type_info: Dict[str, Any]) -> Dict[str, Any]:
        """Resolves typedefs to their underlying concrete types."""
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

    def get_base_type(self, name: str) -> Optional[VtypeBaseType]:
        """Finds a base type (e.g., 'int', 'char') across all loaded ISFs."""
        for f in self._file_order:
            if res := self.vtypejsons[f].get_base_type(name):
                return res
        return None

    def get_user_type(self, name: str) -> Optional[VtypeUserType]:
        """Finds a user-defined struct or union across all loaded ISFs."""
        for f in self._file_order:
            if res := self.vtypejsons[f].get_user_type(name):
                return res
        return None

    def get_enum(self, name: str) -> Optional[VtypeEnum]:
        """Finds an enumeration across all loaded ISFs."""
        for f in self._file_order:
            if res := self.vtypejsons[f].get_enum(name):
                return res
        return None

    def get_symbol(self, name: str) -> Optional[VtypeSymbol]:
        """
        Searches all loaded ISF files for the given symbol name.
        
        Args:
            name: The symbol name to find.
            
        Returns:
            The first valid VtypeSymbol found, or None.
        """
        for path in self._file_order:
            sym = self.vtypejsons[path].get_symbol(name)
            if sym:
                # Skip incomplete symbol definitions if they lack both address and type
                if (not hasattr(sym, 'address') or sym.address in [None, 0]) and not getattr(sym, 'type_info', None):
                    continue
                return sym
        return None
    
    def get_function_address(self, function: str) -> Optional[int]:
        """
        Gets the memory address of a kernel or executable function.
        
        Args:
            function: The function name.
            
        Returns:
            The address as an integer, or None if not found.
        """
        sym = self.get_symbol(function)
        if sym and getattr(sym, 'address', None):
            return sym.address
        return None

    def get_type(self, name: str) -> Optional[Vtype]:
        """General lookup for any type by name."""
        for f in self._file_order:
            if res := self.vtypejsons[f].get_type(name):
                return res
        return None

    def get_symbols_by_address(self, target_address: int) -> List[Any]:
        """Finds all symbols located at a specific memory address."""
        results = []
        for f in self._file_order:
            results.extend(self.vtypejsons[f].get_symbols_by_address(target_address))
        return results

    def get_type_size(self, in_type_info: dict) -> int:
        """Calculates the byte size of a raw ISF type dictionary."""
        type_info = self._resolve_type_info(in_type_info)
        for f in self._file_order:
            res = self.vtypejsons[f].get_type_size(type_info)
            if res is not None:
                return res
        return 0
    
    def _create_instance(
        self,
        type_input: Union[str, Vtype],
        buffer: Union[bytes, bytearray, memoryview],
        instance_offset_in_buffer: int = 0,
    ) -> BoundTypeInstance:
        """
        Internal factory: Creates a BoundTypeInstance by resolving the type across 
        all loaded ISFs and binding it to the provided buffer.
        """
        if isinstance(buffer, bytes):
            processed_buffer = bytearray(buffer)
        elif isinstance(buffer, (bytearray, memoryview)) or hasattr(buffer, "__getitem__"):
            processed_buffer = buffer
        else:
            raise TypeError("Input buffer must be bytes, bytearray, memoryview, or support __getitem__.")

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

        # Bounds checking (skip if using a duck-typed backend proxy)
        if type_def.size is not None and not hasattr(processed_buffer, "backend"):
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

    def shift_symbol_addresses(self, delta: int, path: Optional[str] = None) -> None:
        """
        Shifts the memory addresses of all symbols by a specific delta (useful for ASLR or module loading).
        
        Args:
            delta: The integer offset to add to all symbol addresses.
            path: If specified, only shifts symbols within the targeted ISF path/dict.
        """
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
    
    def _parse_ctype_string_impl(self, ctype: str) -> Union[Vtype, dict, None]:
        """Internal uncached parser for C-style strings."""
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
            return self.get_type(lookup_name)
        elif resolved_info.get("kind") in ("pointer", "array"):
            return resolved_info
        else:
            return self.get_type(resolved_info["name"])

    def typeof(self, ctype: Union[str, Vtype, BoundType, dict]) -> Union[Vtype, dict, None]:
        """
        Resolves a type definition from a string or extracts it from an existing instance.

        Args:
            ctype: A C-style string (e.g., "int *", "struct task_struct"), or an existing DFFI instance.
            
        Returns:
            The resolved Type Definition object or ISF type dictionary.
        """
        if isinstance(ctype, VTYPE_CLASSES):
            return ctype
        if isinstance(ctype, dict):
            return ctype
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
            return self._parse_ctype_string(ctype.strip())
        raise TypeError(
            f"Expected string, BoundTypeInstance, Ptr, or BoundArrayView, got {type(ctype)}"
        )

    def sizeof(self, ctype: Union[str, Vtype, BoundType, Any]) -> int:
        """
        Calculates the memory size in bytes of the given type or instance.
        
        Args:
            ctype: The type name, string, or bound instance.
            
        Returns:
            The size in bytes as an integer.
        """
        if isinstance(ctype, (str, Ptr, BoundArrayView)):
            t = self.typeof(ctype)
        else:
            t = ctype

        size = None
        if isinstance(t, VTYPE_CLASSES):
            return t.size
        elif isinstance(t, BoundTypeInstance):
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
        """
        Returns the absolute offset of the given instance within its underlying host buffer.
        """
        return cdata._instance_offset

    def offsetof(self, ctype: str, *fields_or_indexes: str) -> int:
        """
        Calculates the byte offset of a specific field (or nested path) within a struct/union.

        Args:
            ctype: The name of the root struct or union.
            *fields_or_indexes: A sequence of field names tracking deep into the struct.

        Returns:
            The byte offset from the start of the struct.
        """
        t = self.typeof(ctype)
        if not isinstance(t, VtypeUserType):
            raise TypeError(f"Type '{ctype}' is not a struct or union.")

        offset = 0
        current_type = t

        for field_name in fields_or_indexes:
            if not isinstance(current_type, VtypeUserType):
                raise TypeError(f"Cannot get offset of '{field_name}' inside non-struct type.")

            # Instantly fetch from the O(1) compiled cache
            flat_fields = current_type.get_flattened_fields(self)
            
            if field_name not in flat_fields:
                raise KeyError(f"Type '{current_type.name}' has no field '{field_name}'")

            field, field_offset = flat_fields[field_name]
            offset += field_offset

            # Advance to the next type in the chain
            next_t_info = self._resolve_type_info(field.type_info)
            if next_t_info.get("kind") in ["struct", "union"]:
                current_type = self.get_type(next_t_info.get("name"))
            else:
                current_type = next_t_info

        return offset

    def addressof(self, cdata: BoundTypeInstance, *fields_or_indexes: str) -> Ptr:
        """
        Returns a DFFI Pointer to the given instance, or to a specific nested field within it.
        
        Args:
            cdata: The bound instance to point to.
            *fields_or_indexes: Optional field names to point deep inside the cdata struct.
            
        Returns:
            A `Ptr` object representing the memory address.
        """
        base_addr = cdata._instance_offset
        target_type_info = cdata._instance_type_def

        if fields_or_indexes:
            # 1. Calculate the absolute buffer offset using recursive offsetof
            base_addr += self.offsetof(cdata._instance_type_name, *fields_or_indexes)
            
            # 2. Resolve the final type info using the O(1) flattened fields
            current_type = cdata._instance_type_def
            for field_name in fields_or_indexes:
                if not isinstance(current_type, VtypeUserType):
                    break # Should be caught by offsetof, but safety first
                
                flat_fields = current_type.get_flattened_fields(self)
                if field_name not in flat_fields:
                    break
                    
                field_def, _ = flat_fields[field_name]
                target_type_info = self._resolve_type_info(field_def.type_info)
                
                # If nested, continue the search in the next struct
                if target_type_info.get("kind") in ["struct", "union"]:
                    current_type = self.get_type(target_type_info.get("name"))
                else:
                    current_type = target_type_info

        return Ptr(base_addr, target_type_info, self)

    def _deep_init(self, instance: Any, init: Any) -> None:
        """Recursively initializes complex struct/array instances using standard python lists/dicts."""
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

    def new(self, ctype: Union[str, Vtype, dict], init: Any = None) -> BoundType:
        """
        Allocates a new memory buffer for the specified type and binds it.

        Args:
            ctype: The C-type string or type object (e.g. 'int', 'struct my_struct', 'char[10]', VtypeUserType, VtypeBaseType, VtypeEnum).
            init: Optional initial data (dict, list, string, or bytes) to populate the struct/array.

        Returns:
            The bound, ready-to-use Type Instance or Array View.
        """
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

    def cast(self, ctype: Union[str, Vtype], value: Any) -> BoundType:
        """
        Interprets an integer address or existing CData as a new type.

        Args:
            ctype: The target C-type to cast into (str or type)
            value: The integer memory address, or existing BoundTypeInstance to re-cast.
            
        Returns:
            The newly typed BoundTypeInstance or Ptr.
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
    
    def from_address(self, ctype: Union[str, Vtype, dict], address: int) -> BoundType:
        """
        Creates a new DFFI instance bound to an absolute address in the configured MemoryBackend.
        Operates on LIVE memory via the LiveMemoryProxy.
        """
        if self.backend is None:
            raise RuntimeError("Cannot use from_address(): No memory backend was configured.")

        t = self.typeof(ctype)

        if isinstance(t, dict) and t.get("kind") == "pointer":
            return Ptr(address, t.get("subtype"), self)

        # Wrap the backend in our magic slicing proxy
        proxy = LiveMemoryProxy(self.backend)

        if isinstance(t, dict) and t.get("kind") == "array":
            elem_size = self.sizeof(t.get("subtype")) or 1
            count = t.get("count", 0)
            if count == 0:
                count = (2**63 - 1) // elem_size
                t["count"] = count

            dummy_size = count * elem_size
            dummy_name = f"__dummy_backend_{address}_{hash(str(t))}"
            primary_isf_path = self._file_order[0]
            
            self.vtypejsons[primary_isf_path]._raw_user_types[dummy_name] = {
                "kind": "struct", "size": dummy_size, "fields": {"arr": {"offset": 0, "type": t}}
            }
            self.vtypejsons[primary_isf_path]._parsed_user_types_cache.pop(dummy_name, None)

            instance = self._create_instance(dummy_name, proxy, instance_offset_in_buffer=address)
            return instance.arr

        # If `t` is a dictionary (like when dereferencing a pointer), 
        # resolve it to a concrete type object before passing it to _create_instance!
        if isinstance(t, dict):
            t_name = t.get("name")
            if not t_name:
                raise ValueError(f"Cannot resolve type dictionary missing 'name': {t}")
            t_obj = self.get_type(t_name)
            if t_obj is None:
                raise ValueError(f"Could not resolve concrete type for '{t_name}'")
            t = t_obj

        return self._create_instance(t, proxy, instance_offset_in_buffer=address)

    def from_buffer(
        self,
        ctype: Union[str, Vtype],
        python_buffer: Union[bytearray, memoryview, bytes],
        offset: int = 0,
        require_writable: bool = False,
    ) -> BoundType:
        """
        Creates a new DFFI instance bound directly to an existing Python memory buffer.

        Args:
            ctype: The C-type to interpret the buffer as.
            python_buffer: The host memory (bytearray, bytes, or memoryview).
            offset: The byte offset within the buffer to begin the binding.
            require_writable: If True, raises an error if the host buffer is read-only bytes.
            
        Returns:
            The BoundTypeInstance mapped exactly over the host memory.
        """
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

    def buffer(self, cdata: BoundType, size: int = -1) -> memoryview:
        """
        High-performance extraction: Returns a zero-copy memoryview of the underlying buffer.

        Args:
            cdata: The bound instance to extract memory from.
            size: The number of bytes to capture. Defaults to the C-type's native size.
            
        Returns:
            A python `memoryview` window.
        """
        if isinstance(cdata, Ptr):
            # A pointer doesn't own memory, it just points. 
            # In a real rehosting environment, this would call out to QEMU's memory read API.
            raise TypeError("Cannot get a direct buffer from a Ptr. Dereference it first.")
            
        # Safely extract the buffer and offset based on the view type
        if isinstance(cdata, BoundArrayView):
            buf = cdata._parent_instance._instance_buffer
            offset = cdata._parent_instance._instance_offset + cdata._array_start_offset_in_parent
        else:
            buf = cdata._instance_buffer
            offset = cdata._instance_offset
            
        if size == -1:
            size = self.sizeof(cdata)
            
        # Return a zero-copy slice
        return memoryview(buf)[offset : offset + size]

    def to_bytes(self, cdata: BoundTypeInstance) -> bytes:
        """
        Returns an immutable byte snapshot of the given instance's memory.
        """
        size = cdata._instance_type_def.size
        if size == 0:
            return b""
        return bytes(self.buffer(cdata, size))

    def memmove(
        self,
        dest: Union[BoundTypeInstance, bytearray],
        src: Union[BoundTypeInstance, bytearray, bytes],
        n: int,
    ) -> None:
        """
        Copies memory natively between instances or buffers.
        
        Args:
            dest: Destination buffer or bound instance.
            src: Source buffer or bound instance.
            n: Number of bytes to copy.
        """
        dest_buf = dest._instance_buffer if isinstance(dest, BoundTypeInstance) else dest
        dest_off = dest._instance_offset if isinstance(dest, BoundTypeInstance) else 0

        src_buf = src._instance_buffer if isinstance(src, BoundTypeInstance) else src
        src_off = src._instance_offset if isinstance(src, BoundTypeInstance) else 0

        dest_buf[dest_off : dest_off + n] = src_buf[src_off : src_off + n]

    def string(self, cdata: BoundType, maxlen: int = -1) -> bytes:
        """
        Reads a null-terminated string from memory, or returns the name of an enum value.
        
        Matches CFFI behavior:
        - For enums: returns the name of the enumerator as bytes.
        - For others: reads a null-terminated C-string from the underlying buffer.
        """
        if isinstance(cdata, Ptr):
            raise TypeError("Cannot read string directly from a Ptr.")

        # 1. Handle Enums (CFFI parity)
        # If the instance is an enum, return its constant name
        if isinstance(cdata, BoundTypeInstance) and isinstance(cdata._instance_type_def, VtypeEnum):
            val = cdata._get_value() # Returns EnumInstance
            name = val.name
            if name:
                return name.encode("utf-8")
            return str(int(val)).encode("utf-8") # Fallback to numeric string if name unknown

        # 2. Handle standard memory-based strings
        if isinstance(cdata, BoundArrayView):
            buf = cdata._parent_instance._instance_buffer
            offset = cdata._parent_instance._instance_offset + cdata._array_start_offset_in_parent
        else:
            buf = cdata._instance_buffer
            offset = cdata._instance_offset
        # Handle Live Memory Proxy chunked reads
        if hasattr(buf, "backend"):
            if maxlen > 0:
                byte_data = buf[offset : offset + maxlen]
                null_idx = byte_data.find(b'\x00')
                return byte_data if null_idx == -1 else byte_data[:null_idx]
            
            chunk_size = 64
            result = bytearray()
            current_offset = offset
            while True:
                chunk = buf[current_offset : current_offset + chunk_size]
                if not chunk:
                    break
                null_idx = chunk.find(b'\x00')
                if null_idx != -1:
                    result.extend(chunk[:null_idx])
                    break
                result.extend(chunk)
                current_offset += chunk_size
            return bytes(result)

        max_avail = len(buf) - offset
        read_len = maxlen if maxlen > 0 else max_avail
        
        # Zero-copy slice into Python bytes
        byte_data = memoryview(buf)[offset : offset + read_len].tobytes()
        
        if maxlen > 0:
            return byte_data
            
        # Rapid C-level search for the null terminator
        null_idx = byte_data.find(b'\x00')
        return byte_data if null_idx == -1 else byte_data[:null_idx]

    def unpack(self, cdata: BoundType, count: int = -1) -> Union[list, tuple]:
        """
        Fast-Path API: Dumps a fully primitive struct or array out to Python in a single C operation.

        Args:
            cdata: The array or struct instance to unpack.
            count: Number of elements (for arrays).
            
        Returns:
            A list (for arrays) or tuple (for structs) of extracted values.
        """
        if isinstance(cdata, BoundArrayView):
            if count == -1:
                count = cdata._array_count
            else:
                count = min(count, cdata._array_count)
            # Check if we can fast-path unpack the array using struct
            # We can do this if the array's subtype is a simple base type
            t_info = self._resolve_type_info(cdata._array_subtype_info)
            if t_info.get("kind") == "base":
                base_type = self.get_base_type(t_info.get("name"))
                if base_type:
                    base_struct = base_type.get_compiled_struct()
                    if base_struct:
                        # Build a multiplier format string, e.g., "<1000I"
                        fmt = f"{base_struct.format[0]}{count}{base_struct.format[1:]}"
                        buf = cdata._parent_instance._instance_buffer
                        offset = cdata._parent_instance._instance_offset + cdata._array_start_offset_in_parent
                        return list(struct.unpack_from(fmt, buf, offset))
            
            # Fallback: Array of structs or complex types
            return [cdata[i] for i in range(count)]
            
        # 2. Bulk Struct Unpacking (Fast Path)
        if isinstance(cdata, BoundTypeInstance) and isinstance(cdata._instance_type_def, VtypeUserType):
            agg_struct = cdata._instance_type_def.get_aggregated_struct(self)
            if not agg_struct:
                raise TypeError(
                    f"Struct '{cdata._instance_type_name}' contains complex types, unions, "
                    "or overlapping fields and cannot be bulk-unpacked."
                )
            return agg_struct.unpack_from(cdata._instance_buffer, cdata._instance_offset)

        raise TypeError("unpack() requires an array view or a struct consisting of primitive base types.")

    def cdef(
        self,
        source: str,
        compiler: str = "gcc",
        compiler_flags: Optional[List[str]] = None,
        dwarf2json_cmd: str = "dwarf2json",
        save_isf_to: Optional[str] = None,
    ) -> None:
        """
        Compiles C code on the fly, extracts DWARF info via dwarf2json,
        and heavily loads the resulting types dynamically into this DFFI instance.

        Args:
            source: The raw C code string to compile.
            compiler: The compiler executable.
            compiler_flags: List of flags to pass to the compiler.
            dwarf2json_cmd: Path to the dwarf2json executable.
            save_isf_to: Optional file path to cache the generated ISF.
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
            self._add_vtypejson(vtype_obj, pseudo_path)


    def pretty_print(self, cdata: Any, indent: int = 0, name: str = None) -> str:
        """
        Recursively formats a bound instance, array, or pointer into a human-readable string.
        """
        prefix = "  " * indent
        field_label = f"{name}: " if name else ""

        # Handle Pointers
        if isinstance(cdata, Ptr):
            return f"{prefix}{field_label}{repr(cdata)}"

        # Handle Arrays
        if isinstance(cdata, BoundArrayView):
            count = len(cdata)
            if count == 0:
                return f"{prefix}{field_label}[]"
            if count > 10:
                preview = ", ".join(repr(cdata[i]) for i in range(3))
                return f"{prefix}{field_label}[{preview}, ... ({count} items)]"
            
            res = [f"{prefix}{field_label}["]
            for i in range(count):
                res.append(self.pretty_print(cdata[i], indent + 1))
            res.append(f"{prefix}]")
            return "\n".join(res)

        # Handle Structs/Unions/Base Types
        if isinstance(cdata, BoundTypeInstance):
            type_def = cdata._instance_type_def
            type_name = cdata._instance_type_name
            
            if not isinstance(type_def, VtypeUserType):
                val = cdata._get_value()
                return f"{prefix}{field_label}{val} ({type_name})"

            res = [f"{prefix}{field_label}{type_name} {{"]
            flat_fields = type_def.get_flattened_fields(self)
            for f_name in flat_fields:
                val = getattr(cdata, f_name)
                res.append(self.pretty_print(val, indent + 1, f_name))
            res.append(f"{prefix}}}")
            return "\n".join(res)

        return f"{prefix}{field_label}{repr(cdata)}"

    def to_dict(self, cdata: Any) -> Any:
        """
        Recursively converts a bound instance or array into a standard Python dictionary/list.
        """
        if isinstance(cdata, BoundArrayView):
            return [self.to_dict(item) for item in cdata]

        if isinstance(cdata, BoundTypeInstance):
            type_def = cdata._instance_type_def
            if not isinstance(type_def, VtypeUserType):
                val = cdata._get_value()
                return val._value if hasattr(val, '_value') else val

            res = {}
            flat_fields = type_def.get_flattened_fields(self)
            for f_name in flat_fields:
                val = getattr(cdata, f_name)
                res[f_name] = self.to_dict(val)
            return res

        if isinstance(cdata, Ptr):
            return cdata.address

        return cdata

    def inspect_layout(self, ctype: Union[str, Vtype]) -> None:
        """
        Prints the exact memory layout of a type (pahole-style), showing offsets and padding.
        """
        t = self.typeof(ctype)
        if not isinstance(t, VtypeUserType):
            print(f"Type: {ctype}, Size: {self.sizeof(t)} bytes (Primitive)")
            return

        print(f"Layout of {ctype} (Size: {t.size} bytes):")
        print(f"{'Offset':<8} {'Size':<6} {'Field':<20} {'Type'}")
        print("-" * 60)

        flat_fields = t.get_flattened_fields(self)
        sorted_fields = sorted(flat_fields.items(), key=lambda x: x[1][1])

        last_end = 0
        for f_name, (f_def, abs_offset) in sorted_fields:
            if abs_offset > last_end:
                print(f"{last_end:<8} {abs_offset - last_end:<6} [PADDING]")
            
            f_size = self.get_type_size(f_def.type_info)
            f_type_name = f_def.type_info.get("name") or f_def.type_info.get("kind")
            print(f"{abs_offset:<8} {f_size:<6} {f_name:<20} {f_type_name}")
            last_end = abs_offset + (f_size or 0)

        if last_end < t.size:
            print(f"{last_end:<8} {t.size - last_end:<6} [PADDING]")
    
    def search_symbols(self, pattern: str, use_regex: bool = False) -> Dict[str, Any]:
        """
        Searches for symbols matching a glob or regex pattern.
        
        Args:
            pattern: The glob pattern (e.g., '*sys_call*') or regex string.
            use_regex: If True, evaluates 'pattern' as a regular expression.
        """
        results = {}
        if use_regex:
            compiled = re.compile(pattern)
            for name, sym in self.symbols.items():
                if compiled.search(name):
                    results[name] = sym
        else:
            for name, sym in self.symbols.items():
                if fnmatch.fnmatch(name, pattern):
                    results[name] = sym
        return results

    def search_types(self, pattern: str, use_regex: bool = False) -> Dict[str, "VtypeUserType"]:
        """
        Searches for user types (structs/unions) matching a glob or regex pattern.
        
        Args:
            pattern: The glob pattern (e.g., '*_EPROCESS*') or regex string.
            use_regex: If True, evaluates 'pattern' as a regular expression.
        """
        results = {}
        if use_regex:
            compiled = re.compile(pattern)
            for name, t in self.types.items():
                if compiled.search(name):
                    results[name] = t
        else:
            for name, t in self.types.items():
                if fnmatch.fnmatch(name, pattern):
                    results[name] = t
        return results

    def find_types_with_member(self, member_name: str) -> Dict[str, "VtypeUserType"]:
        """
        Finds all structs/unions that contain a specific member name.
        Very useful for identifying containers of embedded structs (like list nodes).
        
        Args:
            member_name: The exact name of the struct field to search for.
        """
        results = {}
        for name, t in self.types.items():
            if member_name in t.get_flattened_fields(self):
                results[name] = t
        return results