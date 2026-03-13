import io
import lzma
from typing import Any, Dict, List, Optional, Union, cast

import msgspec

from .types import ISFData, VtypeBaseType, VtypeEnum, VtypeMetadata, VtypeSymbol, VtypeUserType


class VtypeJson:
    """
    Parser and container for Intermediate Structure Format (ISF) data.

    This class handles the ingestion of ISF JSON data (from dictionaries, files, or 
    compressed .xz streams) utilizing msgspec for accelerated loading and strict schema 
    enforcement.
    """
    _isf: ISFData

    def __init__(self, isf_input: Union[Dict[str, Any], bytes, str, io.IOBase]):
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
        try:
            if isinstance(isf_input, dict):
                self._isf = msgspec.convert(isf_input, type=ISFData)
            elif isinstance(isf_input, bytes):
                self._isf = msgspec.json.decode(isf_input, type=ISFData)
            elif isinstance(isf_input, str):
                is_xz = isf_input.endswith(".xz")
                if is_xz:
                    with lzma.open(isf_input, "rb") as f:
                        try:
                            file_data = f.read()
                        except lzma.LZMAError as e:
                            raise ValueError(f"Error decompressing XZ file {isf_input}.") from e
                        self._isf = msgspec.json.decode(file_data, type=ISFData)
                else:
                    with open(isf_input, "rb") as f:
                        self._isf = msgspec.json.decode(f.read(), type=ISFData)
            elif hasattr(isf_input, "read"):
                data = isf_input.read()
                if isinstance(data, str):
                    data = data.encode("utf-8")
                self._isf = msgspec.json.decode(data, type=ISFData)
            else:
                raise TypeError(f"Input must be a dict, bytes, file path (str), or file-like object. Got {type(isf_input)}.")
                
        except FileNotFoundError as e:
            raise FileNotFoundError(f"The ISF JSON file was not found: {isf_input!r}") from e
        except (IOError, OSError) as e:
            raise ValueError(f"Could not open or read file '{isf_input!r}'. Error: {e}") from e
        except msgspec.ValidationError as e:
            err_str = str(e)
            if "Expected `object`" in err_str and "got `array`" in err_str:
                raise ValueError("ISF JSON root must be an object, not a list or other type.") from e
            if "missing required field `kind`" in err_str:
                raise ValueError("missing the required 'kind' field (struct, union, etc).") from e
            if "missing required field `base_types`" in err_str or "missing required field `user_types`" in err_str:
                raise ValueError("ISF is missing required top-level sections") from e
            raise ValueError(f"ISF format validation failed: {e}") from e
            
        except msgspec.DecodeError as e:
            raise ValueError(f"Error decoding JSON: {e}") from e

        self.metadata: VtypeMetadata = self._isf.metadata
        self._address_to_symbol_list_cache: Optional[Dict[int, List[VtypeSymbol]]] = None

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
            td = self._isf.typedefs.get(name)
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
        for sym_obj in self._isf.symbols.values():
            if sym_obj is None:
                continue
            addr = getattr(sym_obj, "address", None)
            if addr:
                sym_obj.address = addr + delta
        self._address_to_symbol_list_cache = None

    def get_base_type(self, name: str) -> Optional[VtypeBaseType]:
        """Retrieves a cached VtypeBaseType object by name."""
        return self._isf.base_types.get(name)

    def get_user_type(self, name: str) -> Optional[VtypeUserType]:
        """Retrieves a cached VtypeUserType (struct/union) by name."""
        return self._isf.user_types.get(name)

    def get_enum(self, name: str) -> Optional[VtypeEnum]:
        """Retrieves a cached VtypeEnum object by name."""
        return self._isf.enums.get(name)

    def get_symbol(self, name: str) -> Optional[VtypeSymbol]:
        """Retrieves a cached VtypeSymbol object by name."""
        return self._isf.symbols.get(name)

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
            for symbol_obj in self._isf.symbols.values():
                if symbol_obj is not None and symbol_obj.address is not None:
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
            element_size = self.get_type_size(cast(Dict[str, Any], subtype_info))
            return count * element_size if element_size is not None else None
        if kind == "bitfield":
            # For bitfields, the size is the size of the underlying storage unit type
            underlying_type = type_info.get("type")
            if underlying_type is not None:
                return self.get_type_size(cast(Dict[str, Any], underlying_type))
            return None
        return None

    def __repr__(self) -> str:
        return (
            f"<VtypeJson BaseTypes={len(self._isf.base_types)} UserTypes={len(self._isf.user_types)} "
            f"Enums={len(self._isf.enums)} Symbols={len(self._isf.symbols)}>"
        )
