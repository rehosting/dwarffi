import base64
import struct
from typing import Any, Dict, List, Optional, Tuple


class SourceMetadata:
    """Represents source file metadata within the ISF, tracking provenance."""

    __slots__ = "kind", "name", "hash_type", "hash_value"

    def __init__(self, data: Dict[str, Any]):
        self.kind: Optional[str] = data.get("kind")
        self.name: Optional[str] = data.get("name")
        self.hash_type: Optional[str] = data.get("hash_type")
        self.hash_value: Optional[str] = data.get("hash_value")

    def __repr__(self) -> str:
        return f"<SourceMetadata Name='{self.name}' Kind='{self.kind}'>"


class UnixMetadata:
    """Represents Unix-specific (Linux/Mac) metadata grouping symbols and types."""

    __slots__ = "symbols", "types"

    def __init__(self, data: Dict[str, Any]):
        self.symbols: List[SourceMetadata] = [
            SourceMetadata(s_data) for s_data in data.get("symbols", []) if s_data
        ]
        self.types: List[SourceMetadata] = [
            SourceMetadata(t_data) for t_data in data.get("types", []) if t_data
        ]

    def __repr__(self) -> str:
        return f"<UnixMetadata Symbols={len(self.symbols)} Types={len(self.types)}>"


class VtypeMetadata:
    """Represents the top-level provenance and format metadata in the ISF."""

    __slots__ = "linux", "mac", "producer", "format_version"

    def __init__(self, data: Dict[str, Any]):
        self.linux: Optional[UnixMetadata] = (
            UnixMetadata(data["linux"]) if data.get("linux") else None
        )
        self.mac: Optional[UnixMetadata] = UnixMetadata(data["mac"]) if data.get("mac") else None
        self.producer: Dict[str, str] = data.get("producer", {})
        self.format_version: Optional[str] = data.get("format")

    def __repr__(self) -> str:
        return (
            f"<VtypeMetadata Format='{self.format_version}' Producer='{self.producer.get('name')}'>"
        )


class VtypeBaseType:
    """
    Represents a primitive base type definition (e.g., int, char, float).
    
    Caches a `struct.Struct` object for high-performance memory packing/unpacking.
    """

    __slots__ = "name", "size", "signed", "kind", "endian", "_compiled_struct"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.size: int = data.get("size", 0)
        self.signed: bool = data.get("signed", False)
        self.kind: str = data.get("kind", "int")
        self.endian: str = data.get("endian", "little")
        self._compiled_struct: Optional[struct.Struct] = None

    def get_compiled_struct(self) -> Optional[struct.Struct]:
        """
        Lazily compiles and returns the `struct.Struct` object for this type.
        
        Returns None for types that cannot be packed (like 'void').
        """
        if self._compiled_struct is not None:
            return self._compiled_struct

        if self.size == 0:
            return None

        endian_char = "<" if self.endian == "little" else ">"
        fmt_char: Optional[str] = None

        if self.kind in ("int", "pointer"):
            mapping = {1: "b", 2: "h", 4: "i", 8: "q"}
            fmt_char = mapping.get(self.size)
            if fmt_char and not self.signed:
                fmt_char = fmt_char.upper()
        elif self.kind == "char":
            fmt_char = "b" if self.signed else "B"
        elif self.kind == "bool":
            fmt_char = "?"
        elif self.kind == "float":
            fmt_char = "f" if self.size == 4 else "d"

        if fmt_char:
            try:
                self._compiled_struct = struct.Struct(endian_char + fmt_char)
            except struct.error:
                self._compiled_struct = None
        else:
            self._compiled_struct = None
        return self._compiled_struct

    def __repr__(self) -> str:
        return f"<VtypeBaseType Name='{self.name}' Kind='{self.kind}' Size={self.size} Signed={self.signed}>"


class VtypeStructField:
    """Represents a single field within a user-defined struct or union."""

    __slots__ = "name", "type_info", "offset", "anonymous"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.type_info: Dict[str, Any] = data.get("type", {})
        self.offset: int = data.get("offset", 0)
        self.anonymous: bool = data.get("anonymous", False)

    def __repr__(self) -> str:
        type_kind = self.type_info.get("kind", "unknown")
        type_name_val = self.type_info.get("name", "")
        name_part = f" TypeName='{type_name_val}'" if type_name_val else ""
        return f"<VtypeStructField Name='{self.name}' Offset={self.offset} TypeKind='{type_kind}'{name_part}>"


class VtypeUserType:
    """
    Represents a complex user-defined type (struct or union).
    
    Supports O(1) flattened field lookups and optimized block-unpacking 
    for primitive-only structures.
    """

    __slots__ = "name", "size", "fields", "kind", "_flattened_fields", "_aggregated_struct"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.size: int = data.get("size", 0)
        self.fields: Dict[str, VtypeStructField] = {
            f_name: VtypeStructField(f_name, f_data)
            for f_name, f_data in data.get("fields", {}).items()
            if f_data
        }
        self.kind: str = data.get("kind", "struct")
        self._flattened_fields: Optional[Dict[str, Tuple[VtypeStructField, int]]] = None
        self._aggregated_struct: Optional[struct.Struct] = None

    def get_flattened_fields(self, vtype_accessor: Any) -> Dict[str, Tuple[VtypeStructField, int]]:
        """
        Builds and caches an O(1) lookup table for all fields.
        
        This recursively flattens anonymous unions and structs so their fields
        can be accessed as if they were members of the parent.
        """
        if self._flattened_fields is not None:
            return self._flattened_fields

        flattened: Dict[str, Tuple[VtypeStructField, int]] = {}

        def _flatten(t_def: VtypeUserType, current_offset: int):
            for name, field in t_def.fields.items():
                if not field.anonymous:
                    flattened[name] = (field, current_offset + field.offset)
                else:
                    # Resolve anonymous type and recurse
                    sub_t_info = vtype_accessor._resolve_type_info(field.type_info)
                    sub_t = vtype_accessor.get_type(sub_t_info.get("name"))
                    if isinstance(sub_t, VtypeUserType):
                        _flatten(sub_t, current_offset + field.offset)

        _flatten(self, 0)
        self._flattened_fields = flattened
        return self._flattened_fields

    def get_aggregated_struct(self, vtype_accessor: Any) -> Optional[struct.Struct]:
        """
        Attempts to compile a single `struct.Struct` for the entire object.
        
        This succeeds only if the struct is composed entirely of primitive base types 
        with no overlapping fields (unions) or complex subtypes (arrays/pointers).
        
        Returns a `struct.Struct` for bulk memory unpacking, or None if aggregation is impossible.
        """
        if self._aggregated_struct is not None:
            return self._aggregated_struct
            
        fields_flat = self.get_flattened_fields(vtype_accessor)
        
        # Sort fields by absolute offset to read sequentially
        sorted_fields = sorted(fields_flat.values(), key=lambda x: x[1])
        
        fmt_string = "<" # Assume little endian for the block initially
        current_offset = 0
        
        for field_def, abs_offset in sorted_fields:
            # Overlapping memory (unions) cannot be aggregated into a sequential Struct
            if abs_offset < current_offset:
                return None

            t_info = vtype_accessor._resolve_type_info(field_def.type_info)
            if t_info.get("kind") != "base":
                return None 
                
            base_type = vtype_accessor.get_base_type(t_info.get("name"))
            if not base_type:
                return None

            # Handle compiler padding gaps with 'x'
            if abs_offset > current_offset:
                fmt_string += f"{abs_offset - current_offset}x"
                current_offset = abs_offset
                
            base_struct = base_type.get_compiled_struct()
            if not base_struct:
                return None
                
            fmt_string += base_struct.format[-1]
            current_offset += base_type.size

        try:
            self._aggregated_struct = struct.Struct(fmt_string)
        except struct.error:
            self._aggregated_struct = None
            
        return self._aggregated_struct

    def __repr__(self) -> str:
        return f"<VtypeUserType Name='{self.name}' Kind='{self.kind}' Size={self.size} Fields={len(self.fields)}>"


class VtypeEnum:
    """Represents a C enumeration and its constant mappings."""

    __slots__ = "name", "size", "base", "constants", "_val_to_name"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.size: int = data.get("size", 0)
        self.base: Optional[str] = data.get("base")
        self.constants: Dict[str, int] = data.get("constants", {})
        self._val_to_name: Optional[Dict[int, str]] = None

    def get_name_for_value(self, value: int) -> Optional[str]:
        """Performs a reverse lookup to find a constant name for an integer value."""
        if self._val_to_name is None:
            self._val_to_name = {v: k for k, v in self.constants.items()}
        return self._val_to_name.get(value)

    def __repr__(self) -> str:
        return f"<VtypeEnum Name='{self.name}' Size={self.size} Base='{self.base}' Constants={len(self.constants)}>"


class VtypeSymbol:
    """Represents a global symbol (function or variable) and its memory location."""

    __slots__ = "name", "type_info", "address", "constant_data"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.type_info: Optional[Dict[str, Any]] = data.get("type")
        self.address: Optional[int] = data.get("address")
        self.constant_data: Optional[str] = data.get("constant_data")

    def get_decoded_constant_data(self) -> Optional[bytes]:
        """Decodes base64-encoded constant data associated with the symbol."""
        if self.constant_data:
            try:
                return base64.b64decode(self.constant_data)
            except Exception:
                return None
        return None

    def __repr__(self) -> str:
        type_kind = self.type_info.get("kind", "N/A") if self.type_info else "N/A"
        addr = f"{self.address:#x}" if self.address is not None else "N/A"
        return f"<VtypeSymbol Name='{self.name}' Address={addr} TypeKind='{type_kind}'>"
