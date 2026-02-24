import io
import lzma
from typing import Any, Dict, List, Optional, Union

try:
    import ujson as json

    _JSON_LIB_USED = "ujson"
except ImportError:
    import json

    _JSON_LIB_USED = "json"

from .types import VtypeBaseType, VtypeEnum, VtypeMetadata, VtypeSymbol, VtypeUserType


class VtypeJson:
    """
    Parser and container for Intermediate Structure Format (ISF) data.

    This class handles the ingestion of ISF JSON data (from dictionaries, files, or 
    compressed .xz streams) and provides a lazy-loading interface to resolve 
    Dwarf-derived types and symbols.
    """

    def __init__(self, isf_input: Union[Dict[str, Any], str, io.IOBase]):
        """
        Initializes an ISF definition from a dictionary, file path, or file-like object.

        Args:
            isf_input: ISF data source. Can be a pre-loaded dictionary, a string path 
                       to a .json or .json.xz file, or an open file-like object.

        Raises:
            FileNotFoundError: If a string path is provided but doesn't exist.
            ValueError: If JSON is malformed or required ISF sections are missing.
            TypeError: If input is not one of the supported types.
        """
        raw_data: Dict[str, Any]

        if isinstance(isf_input, dict):
            raw_data = isf_input
        elif isinstance(isf_input, str):
            # Treat string as a file path
            is_xz = isf_input.endswith(".xz")
            try:
                if is_xz:
                    with lzma.open(isf_input, "rt", encoding="utf-8") as f:
                        raw_data = json.load(f)
                else:
                    with open(isf_input, "r", encoding="utf-8") as f:
                        raw_data = json.load(f)
            except FileNotFoundError as e:
                raise FileNotFoundError(f"The ISF JSON file was not found: {isf_input}") from e
            except (IOError, OSError) as e:
                raise ValueError(f"Could not open or read file '{isf_input}'. Error: {e}") from e
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from file {isf_input} (using {_JSON_LIB_USED}).") from e
            except lzma.LZMAError as e:
                raise ValueError(f"Error decompressing XZ file {isf_input}.") from e
        elif hasattr(isf_input, "read"):
            # Treat as a file-like object
            try:
                raw_data = json.load(isf_input)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from file-like object (using {_JSON_LIB_USED}).") from e
        else:
            raise TypeError(f"Input must be a dict, file path (str), or file-like object. Got {type(isf_input)}.")

        if not isinstance(raw_data, dict):
            raise ValueError("ISF JSON root must be an object, not a list or other type.")

        # Basic Schema Validation
        required_sections = ["base_types", "user_types"]
        missing = [s for s in required_sections if s not in raw_data]
        if missing:
            raise ValueError(f"ISF is missing required top-level sections: {missing}")

        # Ensure all user types have a 'kind'
        for name, definition in raw_data.get("user_types", {}).items():
            if "kind" not in definition:
                raise ValueError(f"User type '{name}' is missing the required 'kind' field (struct, union, etc).")

        # Initialize core data structures and metadata
        self.metadata: VtypeMetadata = VtypeMetadata(raw_data.get("metadata", {}))
        self._raw_base_types: Dict[str, Any] = raw_data.get("base_types", {})
        self._parsed_base_types_cache: Dict[str, VtypeBaseType] = {}
        self._raw_user_types: Dict[str, Any] = raw_data.get("user_types", {})
        self._parsed_user_types_cache: Dict[str, VtypeUserType] = {}
        self._raw_enums: Dict[str, Any] = raw_data.get("enums", {})
        self._parsed_enums_cache: Dict[str, VtypeEnum] = {}
        self._raw_symbols: Dict[str, Any] = raw_data.get("symbols", {})
        self._parsed_symbols_cache: Dict[str, VtypeSymbol] = {}
        self._address_to_symbol_list_cache: Optional[Dict[int, List[VtypeSymbol]]] = None
        self._raw_typedefs: Dict[str, Any] = raw_data.get("typedefs", {})

    def _resolve_type_info(self, type_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unrolls C-style typedefs into their underlying target type information.

        Args:
            type_info: A dictionary describing a type (e.g., from a struct field).

        Returns:
            The resolved type dictionary (base, struct, union, pointer, etc).
        """
        visited = set()
        current = type_info
        while current and current.get("kind") == "typedef":
            name = current.get("name")
            if not name:
                break
            if name in visited:
                raise ValueError(f"Circular typedef: {name}")
            visited.add(name)
            td = self._raw_typedefs.get(name)
            if not td:
                break
            current = td
        return current

    def shift_symbol_addresses(self, delta: int) -> None:
        """
        Updates the addresses of all symbols in the ISF by a specific delta.
        Useful for handling ASLR/KASLR or rebased kernel modules.

        Args:
            delta: The integer amount to shift addresses by.
        """
        for _sym_name, sym_data in self._raw_symbols.items():
            if (
                sym_data is not None
                and "address" in sym_data
                and sym_data["address"] not in [None, 0]
            ):
                sym_data["address"] += delta
        for sym_obj in self._parsed_symbols_cache.values():
            if sym_obj.address not in [None, 0]:
                sym_obj.address += delta
        
        # Invalidate the reverse lookup cache after a shift
        self._address_to_symbol_list_cache = None

    def get_base_type(self, name: str) -> Optional[VtypeBaseType]:
        """Retrieves a cached VtypeBaseType object by name."""
        if name in self._parsed_base_types_cache:
            return self._parsed_base_types_cache[name]
        raw_data = self._raw_base_types.get(name)
        if raw_data is None:
            return None
        obj = VtypeBaseType(name, raw_data)
        self._parsed_base_types_cache[name] = obj
        return obj

    def get_user_type(self, name: str) -> Optional[VtypeUserType]:
        """Retrieves a cached VtypeUserType (struct/union) by name."""
        if name in self._parsed_user_types_cache:
            return self._parsed_user_types_cache[name]
        raw_data = self._raw_user_types.get(name)
        if raw_data is None:
            return None
        obj = VtypeUserType(name, raw_data)
        self._parsed_user_types_cache[name] = obj
        return obj

    def get_enum(self, name: str) -> Optional[VtypeEnum]:
        """Retrieves a cached VtypeEnum object by name."""
        if name in self._parsed_enums_cache:
            return self._parsed_enums_cache[name]
        raw_data = self._raw_enums.get(name)
        if raw_data is None:
            return None
        obj = VtypeEnum(name, raw_data)
        self._parsed_enums_cache[name] = obj
        return obj

    def get_symbol(self, name: str) -> Optional[VtypeSymbol]:
        """Retrieves a cached VtypeSymbol object by name."""
        if name in self._parsed_symbols_cache:
            return self._parsed_symbols_cache[name]
        raw_data = self._raw_symbols.get(name)
        if raw_data is None:
            return None
        obj = VtypeSymbol(name, raw_data)
        self._parsed_symbols_cache[name] = obj
        return obj

    def get_type(self, name: str) -> Optional[Union[VtypeUserType, VtypeBaseType, VtypeEnum]]:
        """
        A high-level lookup that resolves a type name, supporting C-style prefixes.

        Args:
            name: Type name, optionally prefixed with 'struct ', 'union ', or 'enum '.

        Returns:
            The resolved type object or None.
        """
        original_name = name
        name_lower = name.lower()

        if name_lower.startswith("struct "):
            return self.get_user_type(original_name[len("struct ") :].strip())
        elif name_lower.startswith("union "):
            user_type = self.get_user_type(original_name[len("union ") :].strip())
            return user_type if user_type and user_type.kind == "union" else None
        elif name_lower.startswith("enum "):
            return self.get_enum(original_name[len("enum ") :].strip())

        return (
            self.get_user_type(original_name)
            or self.get_enum(original_name)
            or self.get_base_type(original_name)
        )

    def get_symbols_by_address(self, target_address: int) -> List[VtypeSymbol]:
        """
        Performs a reverse lookup to find all symbols located at a memory address.
        Initializes a reverse lookup map on the first call.
        """
        if self._address_to_symbol_list_cache is None:
            self._address_to_symbol_list_cache = {}
            for symbol_name in self._raw_symbols.keys():
                symbol_obj = self.get_symbol(symbol_name)
                if symbol_obj and symbol_obj.address is not None:
                    self._address_to_symbol_list_cache.setdefault(symbol_obj.address, []).append(
                        symbol_obj
                    )
        return self._address_to_symbol_list_cache.get(target_address, [])

    def get_type_size(self, in_type_info: Dict[str, Any]) -> Optional[int]:
        """
        Calculates the byte size of a type based on its ISF type dictionary.

        Args:
            in_type_info: Raw ISF type dictionary.

        Returns:
            Total size in bytes or None if the size cannot be determined.
        """
        type_info = self._resolve_type_info(in_type_info)
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
            if not enum_def or not enum_def.base:
                return None
            base_type_for_enum = self.get_base_type(enum_def.base)
            return base_type_for_enum.size if base_type_for_enum else None
        if kind == "array":
            count, subtype_info = type_info.get("count"), type_info.get("subtype")
            if count is None or subtype_info is None:
                return None
            element_size = self.get_type_size(subtype_info)
            return count * element_size if element_size is not None else None
        if kind == "bitfield":
            # For bitfields, the size is the size of the underlying storage unit type
            return self.get_type_size(type_info.get("type")) if type_info.get("type") else None
        return None

    def __repr__(self) -> str:
        return (
            f"<VtypeJson RawBaseTypes={len(self._raw_base_types)} RawUserTypes={len(self._raw_user_types)} "
            f"RawEnums={len(self._raw_enums)} RawSymbols={len(self._raw_symbols)} (Lazy Loaded)>"
        )
