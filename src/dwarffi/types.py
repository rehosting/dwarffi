import base64
import struct
from typing import Any, Dict, List, Optional, Tuple, Union


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


class _FallbackIntStruct:
    """
    A struct.Struct-like duck type for handling exotic integer sizes 
    (e.g., 128-bit or 24-bit integers) that are unsupported by the C struct module.
    """

    __slots__ = ("size", "signed", "endian", "format")

    def __init__(self, size: int, signed: bool, endian: str):
        self.size = size
        self.signed = signed
        self.endian = endian
        self.format = ""  # Empty format aborts sequential aggregation in VtypeUserType

    def unpack_from(self, buffer: Union[bytes, bytearray, memoryview], offset: int = 0) -> Tuple[int]:
        buf_slice = buffer[offset : offset + self.size]
        if len(buf_slice) < self.size:
            buf_slice = bytes(buf_slice) + b'\x00' * (self.size - len(buf_slice))
        return (int.from_bytes(buf_slice, byteorder=self.endian, signed=self.signed),)

    def pack_into(self, buffer: Union[bytearray, memoryview], offset: int, value: int) -> None:
        # Replicate C-style truncation and sign-extension wrapping
        bits = self.size * 8
        mask = (1 << bits) - 1
        val = value & mask
        if self.signed:
            sign_bit = 1 << (bits - 1)
            if val & sign_bit:
                val -= 1 << bits
        
        raw_bytes = val.to_bytes(self.size, byteorder=self.endian, signed=self.signed)
        valid_len = min(self.size, len(buffer) - offset)
        if valid_len > 0:
            buffer[offset : offset + valid_len] = raw_bytes[:valid_len]


class _FallbackBytesStruct:
    """Duck-type for opaque or un-parseable base types (e.g. 80-bit floats, SIMD vectors)."""
    __slots__ = ("size", "format")
    def __init__(self, size: int):
        self.size = size
        self.format = ""

    def unpack_from(self, buffer: Union[bytes, bytearray, memoryview], offset: int = 0) -> Tuple[bytes]:
        buf_slice = buffer[offset : offset + self.size]
        if len(buf_slice) < self.size:
            buf_slice = bytes(buf_slice) + b'\x00' * (self.size - len(buf_slice))
        return (bytes(buf_slice),)

    def pack_into(self, buffer: Union[bytearray, memoryview], offset: int, value: Union[bytes, bytearray]) -> None:
        if not isinstance(value, (bytes, bytearray, memoryview)):
            raise TypeError(f"Unsupported primitive type requires raw bytes, got {type(value).__name__}")
        if len(value) != self.size:
            raise ValueError(f"Expected exactly {self.size} bytes, got {len(value)}")
        valid_len = min(self.size, len(buffer) - offset)
        if valid_len > 0:
            buffer[offset : offset + valid_len] = value[:valid_len]


class VtypeBaseType:
    """
    Represents a primitive base type definition (e.g., int, char, float).
    
    Caches a `struct.Struct` object (or an equivalent duck-type) for 
    high-performance memory packing/unpacking.
    """

    __slots__ = "name", "size", "signed", "kind", "endian", "_compiled_struct"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.size: int = data.get("size", 0)
        self.signed: bool = data.get("signed", False)
        self.kind: str = data.get("kind", "int")
        self.endian: str = data.get("endian", "little")
        self._compiled_struct: Any = None

    def get_compiled_struct(self) -> Any:
        """
        Lazily compiles and returns the `struct.Struct` object for this type.
        Returns a fallback wrapper for arbitrary integer sizes.
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
            if self.size == 2:
                fmt_char = "e"  # IEEE 754 binary16 (half-precision)
            elif self.size == 4:
                fmt_char = "f"  # binary32 (single-precision)
            elif self.size == 8:
                fmt_char = "d"  # binary64 (double-precision)
        elif self.kind == "complex":
            # Natively supported in Python 3.14+, we'll attempt to compile it if present
            if self.size == 8:
                fmt_char = "F"  # float complex
            elif self.size == 16:
                fmt_char = "D"  # double complex

        if fmt_char:
            try:
                self._compiled_struct = struct.Struct(endian_char + fmt_char)
                return self._compiled_struct
            except struct.error:
                # Catch formats unsupported by the running Python version (e.g., F/D in < 3.14)
                pass
 
        # Fallback to pure Python byte manipulation for odd sizes like __int128_t (16 bytes)
        if self.kind in ("int", "pointer"):
            self._compiled_struct = _FallbackIntStruct(self.size, self.signed, self.endian)
            return self._compiled_struct

        # For unhandled floats (80-bit, 128-bit) or SIMD vectors, fall back to raw bytes.
        self._compiled_struct = _FallbackBytesStruct(self.size)
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
        self._flattened_fields: Optional[Dict[str, Tuple[VtypeStructField, int, Dict[str, Any], Any]]] = None
        self._aggregated_struct: Optional[struct.Struct] = None

    def get_flattened_fields(self, vtype_accessor: Any) -> Dict[str, Tuple[VtypeStructField, int, Dict[str, Any], Any]]:
        """
        Builds and caches an O(1) lookup table for all fields.
        Returns a mapping of: field_name -> (field_def, absolute_offset, resolved_type_info, resolved_type_obj)
        This recursively flattens anonymous unions and structs so their fields
        can be accessed as if they were members of the parent.
        """
        if self._flattened_fields is not None:
            return self._flattened_fields

        flattened: Dict[str, Tuple[VtypeStructField, int, Dict[str, Any], Any]] = {}

        def _flatten(t_def: VtypeUserType, current_offset: int):
            for name, field in t_def.fields.items():
                # Pre-resolve typedefs and target type info
                resolved_info = vtype_accessor._resolve_type_info(field.type_info)
                kind = resolved_info.get("kind")
                t_name = resolved_info.get("name")
                
                # Pre-fetch the concrete type object
                resolved_obj = None
                if kind == "base" and t_name:
                    resolved_obj = vtype_accessor.get_base_type(t_name)
                elif kind == "enum" and t_name:
                    resolved_obj = vtype_accessor.get_enum(t_name)
                elif kind in ("struct", "union") and t_name:
                    resolved_obj = vtype_accessor.get_user_type(t_name)

                if not field.anonymous:
                    flattened[name] = (field, current_offset + field.offset, resolved_info, resolved_obj)
                else:
                    # Resolve anonymous type and recurse
                    if isinstance(resolved_obj, VtypeUserType):
                        _flatten(resolved_obj, current_offset + field.offset)

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
        
        # Unpack all 4 elements from the new cache tuple
        for field_def, abs_offset, resolved_info, resolved_obj in sorted_fields:
            # Overlapping memory (unions) cannot be aggregated into a sequential Struct
            if abs_offset < current_offset:
                return None

            # Use the pre-resolved info
            if resolved_info.get("kind") != "base":
                return None 
                
            # Use the pre-resolved object
            base_type = resolved_obj
            if not base_type:
                return None

            # Handle compiler padding gaps with 'x'
            if abs_offset > current_offset:
                fmt_string += f"{abs_offset - current_offset}x"
                current_offset = abs_offset
                
            base_struct = base_type.get_compiled_struct()
            # If base_struct is None, or it's a _FallbackIntStruct lacking a formatting char
            if not base_struct or not base_struct.format:
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

    @property
    def members(self) -> Dict[str, VtypeStructField]:
        """Alias for fields, providing a semantic dictionary of struct/union members."""
        return self.fields

    def to_dict(self) -> Dict[str, Any]:
        """Returns a raw dictionary representation of the type's structure."""
        return {
            "name": self.name,
            "kind": self.kind,
            "size": self.size,
            "fields": {k: {"offset": v.offset, "type": v.type_info, "anonymous": v.anonymous} 
                       for k, v in self.fields.items()}
        }

    def pretty_print(self) -> str:
        """Returns a C-like formatted string of the struct/union layout."""
        lines = [f"{self.kind} {self.name} (size: {self.size} bytes) {{"]
        # Sort fields by offset for a true layout view
        for f_name, f_def in sorted(self.fields.items(), key=lambda x: x[1].offset):
            t_kind = f_def.type_info.get("kind", "unknown")
            t_name = f_def.type_info.get("name", "")
            type_desc = f"{t_kind} {t_name}".strip()
            
            if f_def.anonymous:
                lines.append(f"  [+{f_def.offset:<3}] <anonymous> {type_desc};")
            else:
                lines.append(f"  [+{f_def.offset:<3}] {type_desc} {f_name};")
        lines.append("}")
        return "\n".join(lines)
    
    def __str__(self) -> str:
        return self.pretty_print()


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
    
    @property
    def members(self) -> Dict[str, int]:
        """Alias for constants, providing a semantic dictionary of enum members."""
        return self.constants

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "base": self.base,
            "constants": self.constants
        }

    def pretty_print(self) -> str:
        """Returns a C-like formatted string of the enum layout."""
        lines = [f"enum {self.name} (size: {self.size}, base: {self.base}) {{"]
        for name, val in sorted(self.constants.items(), key=lambda x: x[1]):
            lines.append(f"  {name} = {val},")
        lines.append("}")
        return "\n".join(lines)
        
    def __str__(self) -> str:
        return self.pretty_print()


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
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "address": self.address,
            "type_info": self.type_info
        }

    def pretty_print(self) -> str:
        addr_str = f"{self.address:#x}" if self.address is not None else "N/A"
        
        if self.type_info:
            kind = self.type_info.get('kind', '')
            name = self.type_info.get('name', '')
            
            # Arrays and pointers nest their target type name under 'subtype'
            if not name and 'subtype' in self.type_info:
                name = self.type_info['subtype'].get('name', '')
                
            type_str = f"{kind} {name}".strip()
        else:
            type_str = "unknown"
            
        return f"Symbol {self.name} @ {addr_str} (Type: {type_str})"

    def __str__(self) -> str:
        return self.pretty_print()