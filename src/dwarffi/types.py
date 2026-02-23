import base64
import struct
from typing import Any, Dict, List, Optional


class SourceMetadata:
    """Represents source file metadata within the ISF."""

    __slots__ = "kind", "name", "hash_type", "hash_value"

    def __init__(self, data: Dict[str, Any]):
        self.kind: Optional[str] = data.get("kind")
        self.name: Optional[str] = data.get("name")
        self.hash_type: Optional[str] = data.get("hash_type")
        self.hash_value: Optional[str] = data.get("hash_value")

    def __repr__(self) -> str:
        return f"<SourceMetadata Name='{self.name}' Kind='{self.kind}'>"


class UnixMetadata:
    """Represents Unix-specific (Linux/Mac) metadata within the ISF."""

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
    """Represents the top-level metadata in the ISF."""

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
    """Represents a base type definition in the ISF (e.g., int, char)."""

    __slots__ = "name", "size", "signed", "kind", "endian", "_compiled_struct"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.size: Optional[int] = data.get("size")
        self.signed: Optional[bool] = data.get("signed")
        self.kind: Optional[str] = data.get("kind")
        self.endian: Optional[str] = data.get("endian")
        self._compiled_struct: Optional[struct.Struct] = None

    def get_compiled_struct(self) -> Optional[struct.Struct]:
        if hasattr(self, "_compiled_struct") and self._compiled_struct is not None:
            if self.size == 0 and self._compiled_struct is None:
                return None
            if self.size != 0:
                return self._compiled_struct
        elif self.size == 0:
            self._compiled_struct = None
            return None

        if self.size is None or self.kind is None or self.endian is None:
            return None
        if self.size == 0 and self.kind == "void":
            self._compiled_struct = None
            return None

        endian_char = "<" if self.endian == "little" else ">"
        fmt_char: Optional[str] = None

        if self.kind == "int" or self.kind == "pointer":
            if self.size == 1:
                fmt_char = "b" if self.signed else "B"
            elif self.size == 2:
                fmt_char = "h" if self.signed else "H"
            elif self.size == 4:
                fmt_char = "i" if self.signed else "I"
            elif self.size == 8:
                fmt_char = "q" if self.signed else "Q"
        elif self.kind == "char":
            if self.size == 1:
                fmt_char = "b" if self.signed else "B"
        elif self.kind == "bool":
            if self.size == 1:
                fmt_char = "?"
        elif self.kind == "float":
            if self.size == 4:
                fmt_char = "f"
            elif self.size == 8:
                fmt_char = "d"

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
    """Represents a field within a user-defined struct or union."""

    __slots__ = "name", "type_info", "offset", "anonymous"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.type_info: Dict[str, Any] = data.get("type", {})
        self.offset: Optional[int] = data.get("offset")
        self.anonymous: Optional[bool] = data.get("anonymous", False)

    def __repr__(self) -> str:
        type_kind = self.type_info.get("kind", "unknown")
        type_name_val = self.type_info.get("name", "")
        name_part = f" TypeName='{type_name_val}'" if type_name_val else ""
        return f"<VtypeStructField Name='{self.name}' Offset={self.offset} TypeKind='{type_kind}'{name_part}>"


class VtypeUserType:
    """Represents a user-defined type (struct or union) in the ISF."""

    __slots__ = "name", "size", "fields", "kind"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.size: Optional[int] = data.get("size")
        self.fields: Dict[str, VtypeStructField] = {
            f_name: VtypeStructField(f_name, f_data)
            for f_name, f_data in data.get("fields", {}).items()
            if f_data
        }
        self.kind: Optional[str] = data.get("kind")  # "struct" or "union"

    def __repr__(self) -> str:
        return f"<VtypeUserType Name='{self.name}' Kind='{self.kind}' Size={self.size} Fields={len(self.fields)}>"


class VtypeEnum:
    """Represents an enumeration type in the ISF."""

    __slots__ = "name", "size", "base", "constants", "_val_to_name"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.size: Optional[int] = data.get("size")
        self.base: Optional[str] = data.get("base")
        self.constants: Dict[str, int] = data.get("constants", {})
        self._val_to_name: Optional[Dict[int, str]] = None

    def get_name_for_value(self, value: int) -> Optional[str]:
        if self._val_to_name is None:
            self._val_to_name = {v: k for k, v in self.constants.items()}
        return self._val_to_name.get(value)

    def __repr__(self) -> str:
        return f"<VtypeEnum Name='{self.name}' Size={self.size} Base='{self.base}' Constants={len(self.constants)}>"


class VtypeSymbol:
    """Represents a symbol (variable or function) in the ISF."""

    __slots__ = "name", "type_info", "address", "constant_data"

    def __init__(self, name: str, data: Dict[str, Any]):
        self.name: str = name
        self.type_info: Optional[Dict[str, Any]] = data.get("type")
        self.address: Optional[int] = data.get("address")
        self.constant_data: Optional[str] = data.get("constant_data")

    def get_decoded_constant_data(self) -> Optional[bytes]:
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
