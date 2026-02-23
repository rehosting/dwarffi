import io
import lzma
from typing import Any, Dict, List, Optional, Union

try:
    import ujson as json
    _JSON_LIB_USED = "ujson"
except ImportError:
    import json
    _JSON_LIB_USED = "json"

from .types import (
    VtypeMetadata, VtypeBaseType, VtypeUserType,
    VtypeEnum, VtypeSymbol
)
from .instances import BoundTypeInstance

class VtypeJson:
    def __init__(self, data: Dict[str, Any]):
        self.metadata: VtypeMetadata = VtypeMetadata(data.get("metadata", {}))
        self._raw_base_types: Dict[str, Any] = data.get("base_types", {})
        self._parsed_base_types_cache: Dict[str, VtypeBaseType] = {}
        self._raw_user_types: Dict[str, Any] = data.get("user_types", {})
        self._parsed_user_types_cache: Dict[str, VtypeUserType] = {}
        self._raw_enums: Dict[str, Any] = data.get("enums", {})
        self._parsed_enums_cache: Dict[str, VtypeEnum] = {}
        self._raw_symbols: Dict[str, Any] = data.get("symbols", {})
        self._parsed_symbols_cache: Dict[str, VtypeSymbol] = {}
        self._address_to_symbol_list_cache: Optional[Dict[int, List[VtypeSymbol]]] = None
        self._raw_typedefs: Dict[str, Any] = data.get("typedefs", {})
    
    def resolve_type_info(self, type_info: Dict[str, Any]) -> Dict[str, Any]:
        """Unrolls typedefs into their underlying target type info."""
        visited = set()
        current = type_info
        while current and current.get("kind") == "typedef":
            name = current.get("name")
            if not name: break
            if name in visited:
                raise ValueError(f"Circular typedef: {name}")
            visited.add(name)
            td = self._raw_typedefs.get(name)
            if not td: break
            current = td
        return current

    def shift_symbol_addresses(self, delta: int):
        for _sym_name, sym_data in self._raw_symbols.items():
            if sym_data is not None and "address" in sym_data and sym_data["address"] not in [None, 0]:
                sym_data["address"] += delta
        for sym_obj in self._parsed_symbols_cache.values():
            if sym_obj.address not in [None, 0]:
                sym_obj.address += delta
        self._address_to_symbol_list_cache = None

    def get_base_type(self, name: str) -> Optional[VtypeBaseType]:
        if name in self._parsed_base_types_cache:
            return self._parsed_base_types_cache[name]
        raw_data = self._raw_base_types.get(name)
        if raw_data is None: return None
        obj = VtypeBaseType(name, raw_data)
        self._parsed_base_types_cache[name] = obj
        return obj

    def get_user_type(self, name: str) -> Optional[VtypeUserType]:
        if name in self._parsed_user_types_cache:
            return self._parsed_user_types_cache[name]
        raw_data = self._raw_user_types.get(name)
        if raw_data is None: return None
        obj = VtypeUserType(name, raw_data)
        self._parsed_user_types_cache[name] = obj
        return obj

    def get_enum(self, name: str) -> Optional[VtypeEnum]:
        if name in self._parsed_enums_cache:
            return self._parsed_enums_cache[name]
        raw_data = self._raw_enums.get(name)
        if raw_data is None: return None
        obj = VtypeEnum(name, raw_data)
        self._parsed_enums_cache[name] = obj
        return obj

    def get_symbol(self, name: str) -> Optional[VtypeSymbol]:
        if name in self._parsed_symbols_cache:
            return self._parsed_symbols_cache[name]
        raw_data = self._raw_symbols.get(name)
        if raw_data is None: return None
        obj = VtypeSymbol(name, raw_data)
        self._parsed_symbols_cache[name] = obj
        return obj

    def get_type(self, name: str) -> Optional[Union[VtypeUserType, VtypeBaseType, VtypeEnum]]:
        original_name = name
        name_lower = name.lower()

        if name_lower.startswith("struct "):
            return self.get_user_type(original_name[len("struct "):].strip())
        elif name_lower.startswith("union "):
            user_type = self.get_user_type(original_name[len("union "):].strip())
            return user_type if user_type and user_type.kind == "union" else None
        elif name_lower.startswith("enum "):
            return self.get_enum(original_name[len("enum "):].strip())

        return self.get_user_type(original_name) or self.get_enum(original_name) or self.get_base_type(original_name)

    def get_symbols_by_address(self, target_address: int) -> List[VtypeSymbol]:
        if self._address_to_symbol_list_cache is None:
            self._address_to_symbol_list_cache = {}
            for symbol_name in self._raw_symbols.keys():
                symbol_obj = self.get_symbol(symbol_name)
                if symbol_obj and symbol_obj.address is not None:
                    self._address_to_symbol_list_cache.setdefault(symbol_obj.address, []).append(symbol_obj)
        return self._address_to_symbol_list_cache.get(target_address, [])

    def get_type_size(self, in_type_info: Dict[str, Any]) -> Optional[int]:
        type_info = self.resolve_type_info(in_type_info)
        kind, name = type_info.get("kind"), type_info.get("name")
        if kind == "base":
            base_def = self.get_base_type(name) if name else None
            return base_def.size if base_def else None
        if kind == "pointer":
            ptr_base_def = self.get_base_type("pointer")
            return ptr_base_def.size if ptr_base_def else None
        if kind in ["struct", "union"]:
            user_def = self.get_user_type(name) if name else None
            return user_def.size if user_def else None
        if kind == "enum":
            enum_def = self.get_enum(name) if name else None
            if not enum_def or not enum_def.base: return None
            base_type_for_enum = self.get_base_type(enum_def.base)
            return base_type_for_enum.size if base_type_for_enum else None
        if kind == "array":
            count, subtype_info = type_info.get("count"), type_info.get("subtype")
            if None in [count, subtype_info]: return None
            element_size = self.get_type_size(subtype_info)
            return count * element_size if element_size is not None else None
        if kind == "bitfield":
            return self.get_type_size(type_info.get("type")) if type_info.get("type") else None
        return None

    def create_instance(self, type_input: Union[str, VtypeUserType, VtypeBaseType, VtypeEnum],
                        buffer: Union[bytes, bytearray],
                        instance_offset_in_buffer: int = 0) -> BoundTypeInstance:
        type_def: Optional[Union[VtypeUserType, VtypeBaseType, VtypeEnum]] = None
        type_name_for_instance: str

        if isinstance(buffer, bytes):
            processed_buffer = bytearray(buffer)
        elif isinstance(buffer, bytearray):
            processed_buffer = buffer
        else:
            raise TypeError("Input buffer for create_instance must be bytes or bytearray.")

        if isinstance(type_input, str):
            type_name_for_instance = type_input
            type_def = self.get_type(type_input)
        elif isinstance(type_input, (VtypeUserType, VtypeBaseType, VtypeEnum)):
            type_def = type_input
            type_name_for_instance = type_def.name
        else:
            raise TypeError(f"type_input must be a string or Vtype object, got {type(type_input)}")

        if type_def:
            if not hasattr(type_def, 'size') or type_def.size is None:
                type_kind = type_def.kind if hasattr(type_def, "kind") else None
                if not (type_kind == 'void' and type_def.size == 0):
                    raise ValueError(f"Type definition for '{type_name_for_instance}' lacks a valid size.")

            if type_def.size is not None:
                effective_len = len(processed_buffer) - instance_offset_in_buffer
                if type_def.size > effective_len:
                    raise ValueError(f"Buffer too small for '{type_name_for_instance}' at offset {instance_offset_in_buffer}.")
            return BoundTypeInstance(type_name_for_instance, type_def, processed_buffer, self, instance_offset_in_buffer)

        raise ValueError(f"Type definition for '{type_input if isinstance(type_input, str) else type_input.name}' not found.")

    def __repr__(self) -> str:
        return (f"<VtypeJson RawBaseTypes={len(self._raw_base_types)} RawUserTypes={len(self._raw_user_types)} "
                f"RawEnums={len(self._raw_enums)} RawSymbols={len(self._raw_symbols)} (Lazy Loaded)>")

class VtypeJsonGroup:
    def __init__(self, file_list: list):
        self._file_order = list(file_list)
        self.vtypejsons = {}
        for f in self._file_order:
            self.vtypejsons[f] = load_isf_json(f)

    @property
    def paths(self):
        return list(self._file_order)
    
    def resolve_type_info(self, type_info: Dict[str, Any]) -> Dict[str, Any]:
        visited = set()
        current = type_info
        while current and current.get("kind") == "typedef":
            name = current.get("name")
            if not name: break
            if name in visited:
                raise ValueError(f"Circular typedef: {name}")
            visited.add(name)
            
            td = None
            for f in self._file_order:
                td = self.vtypejsons[f]._raw_typedefs.get(name)
                if td: break
            if not td: break
            current = td
        return current

    def get_vtypejson(self, path): return self.vtypejsons[path]

    def get_base_type(self, name: str):
        for f in self._file_order:
            if res := self.vtypejsons[f].get_base_type(name): return res

    def get_user_type(self, name: str):
        for f in self._file_order:
            if res := self.vtypejsons[f].get_user_type(name): return res

    def get_enum(self, name: str):
        for f in self._file_order:
            if res := self.vtypejsons[f].get_enum(name): return res

    def get_symbol(self, name: str):
        for f in self._file_order:
            res = self.vtypejsons[f].get_symbol(name)
            if res and not (hasattr(res, 'address') and res.address in [None, 0]):
                return res

    def get_type(self, name: str):
        for f in self._file_order:
            if res := self.vtypejsons[f].get_type(name): return res

    def get_symbols_by_address(self, target_address: int):
        results = []
        for f in self._file_order:
            results.extend(self.vtypejsons[f].get_symbols_by_address(target_address))
        return results

    def get_type_size(self, in_type_info: dict):
        type_info = self.resolve_type_info(in_type_info)
        for f in self._file_order:
            if res := self.vtypejsons[f].get_type_size(type_info): return res

    def create_instance(self, type_input, buffer, instance_offset_in_buffer=0):
        for f in self._file_order:
            try:
                return self.vtypejsons[f].create_instance(type_input, buffer, instance_offset_in_buffer)
            except ValueError:
                continue
        raise ValueError(f"Type definition not found in any VtypeJson.")

    def shift_symbol_addresses(self, delta: int, path: str = None):
        if path is None:
            for f in self._file_order:
                self.vtypejsons[f].shift_symbol_addresses(delta)
        else:
            self.vtypejsons[path].shift_symbol_addresses(delta)

    def __repr__(self):
        return f"<VtypeJsonGroup: {len(self.vtypejsons)} VtypeJsons>"

def load_isf_json(json_input: Union[str, object]) -> VtypeJson:
    raw_data: Any
    input_is_path_str = isinstance(json_input, str)
    if input_is_path_str:
        path_str = str(json_input)
        is_xz = path_str.endswith(".xz")
        try:
            if is_xz:
                with lzma.open(path_str, 'rt', encoding='utf-8') as f:
                    raw_data = json.load(f)
            else:
                with open(path_str, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The ISF JSON file was not found: {path_str}") from e
        except (IOError, OSError) as e:
            raise ValueError(f"Could not open or read file '{path_str}'. Error: {e}") from e
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from file {path_str} (using {_JSON_LIB_USED}).") from e
        except lzma.LZMAError as e:
            raise ValueError(f"Error decompressing XZ file {path_str}.") from e
    elif hasattr(json_input, 'read'):
        try:
            raw_data = json.load(json_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from file-like object (using {_JSON_LIB_USED}).") from e
    else:
        raise TypeError(f"Input must be a JSON string (path or content), or a file-like object. Got {type(json_input)}.")
    
    if not isinstance(raw_data, dict):
        raise ValueError("ISF JSON root must be an object, not a list or other type.")
    return VtypeJson(raw_data)

def isf_from_dict(isf_dict: dict[str, Any]) -> VtypeJson:
    f = io.StringIO(json.dumps(isf_dict))
    return load_isf_json(f)