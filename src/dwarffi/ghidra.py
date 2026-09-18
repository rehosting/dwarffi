"""Snapshot Ghidra programs into DFFI's ISF dictionary format.

This module intentionally avoids importing Ghidra or PyGhidra at import time.
The converter uses the small subset of the Ghidra Java API that is also easy to
fake in unit tests, so regular dwarffi usage does not gain a Ghidra dependency.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, List, Optional, Set

FORMAT_VERSION = "6.2.0"
TOOL_NAME = "pyghidra2isf"
TOOL_VERSION = "0.1.0"


def current_program_from_context() -> Any:
    """Return a PyGhidra/GhidraScript ``currentProgram`` from the call stack."""
    for frame_info in inspect.stack()[1:]:
        frame = frame_info.frame
        if "currentProgram" in frame.f_locals:
            return frame.f_locals["currentProgram"]
        if "currentProgram" in frame.f_globals:
            return frame.f_globals["currentProgram"]
    raise RuntimeError(
        "No active Ghidra program was found. Call DFFI.from_ghidra(currentProgram) "
        "from your PyGhidra or GhidraScript context."
    )


def program_to_isf(
    program: Any,
    *,
    include_symbols: bool = True,
    include_functions: bool = True,
    types_only: bool = False,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an ISF dictionary snapshot from a Ghidra ``Program``."""
    if program is None:
        raise ValueError("program must be a Ghidra Program object, not None")
    exporter = _ProgramExporter(program, source_name=source_name)
    return exporter.export(
        include_symbols=include_symbols and not types_only,
        include_functions=include_functions and not types_only,
    )


class _ProgramExporter:
    def __init__(self, program: Any, *, source_name: Optional[str] = None) -> None:
        self.program = program
        self.source_name = source_name or str(_call(program, "getName", default="ghidra_program"))
        language = _call(program, "getLanguage", default=None)
        is_big = bool(_call(language, "isBigEndian", default=False)) if language is not None else False
        self.endian = "big" if is_big else "little"
        self.pointer_size = max(int(_call(program, "getDefaultPointerSize", default=8) or 8), 1)

        self.base_types: Dict[str, Any] = {}
        self.user_types: Dict[str, Any] = {}
        self.enums: Dict[str, Any] = {}
        self.typedefs: Dict[str, Any] = {}
        self.symbols: Dict[str, Any] = {}
        self.functions: Dict[str, Any] = {}

    def export(self, *, include_symbols: bool, include_functions: bool) -> Dict[str, Any]:
        self._export_types()
        if include_symbols:
            self._export_symbols()
        if include_functions:
            self._export_functions()
        return {
            "metadata": self._metadata(),
            "base_types": dict(sorted(self.base_types.items())),
            "user_types": dict(sorted(self.user_types.items())),
            "enums": dict(sorted(self.enums.items())),
            "symbols": dict(sorted(self.symbols.items())),
            "functions": dict(sorted(self.functions.items())),
            "typedefs": dict(sorted(self.typedefs.items())),
        }

    def _metadata(self) -> Dict[str, Any]:
        source = {"kind": "ghidra_program", "name": self.source_name}
        return {
            "producer": {
                "name": TOOL_NAME,
                "version": TOOL_VERSION,
                "ghidra_version": _ghidra_version(),
            },
            "format": FORMAT_VERSION,
            "ghidra": {"types": [source], "symbols": [source]},
        }

    def _export_types(self) -> None:
        self._ensure_void()
        self._ensure_pointer()
        manager = _call(self.program, "getDataTypeManager", default=None)
        for data_type in _iter(_call(manager, "getAllDataTypes", default=[])):
            if data_type is None or _is_default(data_type):
                continue
            self._export_data_type(data_type)

    def _export_data_type(self, data_type: Any) -> None:
        if _is_structure(data_type):
            self._export_composite(data_type, "struct")
        elif _is_union(data_type):
            self._export_composite(data_type, "union")
        elif _is_enum(data_type):
            self._export_enum(data_type)
        elif _is_typedef(data_type):
            self._export_typedef(data_type)
        elif _is_pointer(data_type):
            self._ensure_pointer()
            self._type_ref(_call(data_type, "getDataType", default=None))
        elif _is_array(data_type):
            self._type_ref(_call(data_type, "getDataType", default=None))
        elif _is_function_definition(data_type):
            return
        else:
            self._ensure_base(data_type)

    def _export_composite(self, data_type: Any, kind: str) -> None:
        name = _type_name(data_type)
        if name in self.user_types:
            return

        fields: Dict[str, Any] = {}
        anonymous_count = 0
        for component in _iter(_call(data_type, "getComponents", default=[])):
            field_type = _call(component, "getDataType", default=None)
            if field_type is None or _is_default(field_type):
                continue
            field_name = _call(component, "getFieldName", default=None)
            anonymous = field_name is None or str(field_name) == ""
            if anonymous:
                field_name = f"unnamed_field_{anonymous_count}"
                anonymous_count += 1

            field = {
                "type": self._field_type_ref(component),
                "offset": max(int(_call(component, "getOffset", default=0) or 0), 0),
            }
            if anonymous:
                field["anonymous"] = True
            fields[str(field_name)] = field

        self.user_types[name] = {
            "size": max(int(_call(data_type, "getLength", default=0) or 0), 0),
            "fields": dict(sorted(fields.items())),
            "kind": kind,
        }

    def _field_type_ref(self, component: Any) -> Dict[str, Any]:
        data_type = _call(component, "getDataType", default=None)
        if _is_bitfield(data_type):
            return {
                "kind": "bitfield",
                "bit_length": _first_int(data_type, 0, "getDeclaredBitSize", "getBitSize"),
                "bit_position": _first_int(component, 0, "getBitOffset"),
                "type": self._type_ref(_call(data_type, "getBaseDataType", default=None)),
            }
        return self._type_ref(data_type)

    def _export_enum(self, enum_type: Any) -> None:
        name = _type_name(enum_type)
        if name in self.enums:
            return
        constants = {
            str(const_name): int(_call(enum_type, "getValue", const_name, default=0) or 0)
            for const_name in _iter(_call(enum_type, "getNames", default=[]))
        }
        self.enums[name] = {
            "size": max(int(_call(enum_type, "getLength", default=0) or 0), 0),
            "base": self._enum_base_name(enum_type, constants),
            "constants": dict(sorted(constants.items())),
        }

    def _enum_base_name(self, enum_type: Any, constants: Dict[str, int]) -> str:
        length = max(int(_call(enum_type, "getLength", default=0) or 0), 0)
        signed = any(value < 0 for value in constants.values())
        name = f"{'int' if signed else 'uint'}{length * 8}_t"
        self._ensure_synthetic_base(name, length, "int", signed)
        return name

    def _export_typedef(self, type_def: Any) -> None:
        name = _type_name(type_def)
        if name not in self.typedefs:
            self.typedefs[name] = self._type_ref(_call(type_def, "getBaseDataType", default=None))

    def _export_symbols(self) -> None:
        symbol_table = _call(self.program, "getSymbolTable", default=None)
        listing = _call(self.program, "getListing", default=None)
        memory = _call(self.program, "getMemory", default=None)
        for symbol in _iter(_call(symbol_table, "getAllSymbols", True, default=[])):
            if symbol is None or bool(_call(symbol, "isExternal", default=False)):
                continue
            if _is_function_symbol(symbol):
                continue
            address = _call(symbol, "getAddress", default=None)
            if address is None:
                continue
            if memory is not None and not bool(_call(memory, "contains", address, default=True)):
                continue

            record: Dict[str, Any] = {"address": _address_offset(address)}
            data = _call(listing, "getDataAt", address, default=None) if listing is not None else None
            data_type = _call(data, "getDataType", default=None)
            if data_type is not None:
                record["type"] = self._type_ref(data_type)
            self.symbols[str(_call(symbol, "getName", True, default=_call(symbol, "getName", default="")))] = record

    def _export_functions(self) -> None:
        listing = _call(self.program, "getListing", default=None)
        for function in _iter(_call(listing, "getFunctions", True, default=[])):
            if function is None or bool(_call(function, "isExternal", default=False)):
                continue
            parameters: List[Dict[str, Any]] = []
            for parameter in _iter(_call(function, "getParameters", default=[])):
                parameters.append(
                    {
                        "name": str(_call(parameter, "getName", default="")),
                        "type": self._type_ref(_call(parameter, "getDataType", default=None)),
                    }
                )
            self.functions[
                str(_call(function, "getName", True, default=_call(function, "getName", default="")))
            ] = {
                "address": _address_offset(_call(function, "getEntryPoint", default=0)),
                "return_type": self._type_ref(_call(function, "getReturnType", default=None)),
                "parameters": parameters,
            }

    def _type_ref(self, data_type: Any) -> Dict[str, Any]:
        if data_type is None or _is_void(data_type):
            self._ensure_void()
            return {"kind": "base", "name": "void"}
        if _is_typedef(data_type):
            self._export_typedef(data_type)
            return {"kind": "typedef", "name": _type_name(data_type)}
        if _is_pointer(data_type):
            self._ensure_pointer()
            return {
                "kind": "pointer",
                "subtype": self._type_ref(_call(data_type, "getDataType", default=None)),
            }
        if _is_array(data_type):
            return {
                "kind": "array",
                "count": max(int(_call(data_type, "getNumElements", default=0) or 0), 0),
                "subtype": self._type_ref(_call(data_type, "getDataType", default=None)),
            }
        if _is_structure(data_type):
            self._export_composite(data_type, "struct")
            return {"kind": "struct", "name": _type_name(data_type)}
        if _is_union(data_type):
            self._export_composite(data_type, "union")
            return {"kind": "union", "name": _type_name(data_type)}
        if _is_enum(data_type):
            self._export_enum(data_type)
            return {"kind": "enum", "name": _type_name(data_type)}
        if _is_function_definition(data_type):
            return {
                "kind": "function",
                "return_type": self._type_ref(_call(data_type, "getReturnType", default=None)),
                "parameters": [
                    {
                        "name": str(_call(param, "getName", default="")),
                        "type": self._type_ref(_call(param, "getDataType", default=None)),
                    }
                    for param in _iter(_call(data_type, "getArguments", default=[]))
                ],
            }

        return {"kind": "base", "name": self._ensure_base(data_type)}

    def _ensure_base(self, data_type: Any) -> str:
        if data_type is None or _is_void(data_type):
            self._ensure_void()
            return "void"
        name = _type_name(data_type)
        size = max(int(_call(data_type, "getLength", default=0) or 0), 0)
        if size == 0 and name != "void":
            name = "opaque_0"
        self._ensure_synthetic_base(name, size, _base_kind(data_type), _is_signed_base(data_type))
        return name

    def _ensure_void(self) -> None:
        self._ensure_synthetic_base("void", 0, "void", False)

    def _ensure_pointer(self) -> None:
        self._ensure_synthetic_base("pointer", self.pointer_size, "pointer", False)

    def _ensure_synthetic_base(self, name: str, size: int, kind: str, signed: bool) -> None:
        if name not in self.base_types:
            self.base_types[name] = {
                "size": max(size, 0),
                "signed": signed,
                "kind": kind,
                "endian": self.endian,
            }


def _call(obj: Any, method: str, *args: Any, default: Any = None) -> Any:
    if obj is None:
        return default
    attr = getattr(obj, method, None)
    if attr is None:
        return default
    if not callable(attr):
        return attr
    try:
        return attr(*args)
    except TypeError:
        if args:
            try:
                return attr()
            except TypeError:
                return default
        return default


def _iter(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if hasattr(value, "hasNext") and hasattr(value, "next"):
        return _java_iterator(value)
    return value


def _java_iterator(value: Any) -> Iterable[Any]:
    while bool(_call(value, "hasNext", default=False)):
        yield _call(value, "next", default=None)


def _class_names(obj: Any) -> List[str]:
    names = [cls.__name__ for cls in type(obj).__mro__]
    java_class = _call(obj, "getClass", default=None)
    names.extend(_java_class_names(java_class))
    return names


def _has_class(obj: Any, name: str) -> bool:
    target = name.lower()
    return any(
        target == cls_name.lower()
        or cls_name.lower().endswith(target)
        or target in cls_name.lower()
        for cls_name in _class_names(obj)
    )


def _java_class_names(java_class: Any, seen: Optional[Set[int]] = None) -> List[str]:
    if java_class is None:
        return []
    if seen is None:
        seen = set()
    identity = id(java_class)
    if identity in seen:
        return []
    seen.add(identity)

    names: List[str] = []
    for method in ("getSimpleName", "getName"):
        value = _call(java_class, method, default=None)
        if value:
            text = str(value)
            names.append(text)
            names.append(text.split(".")[-1])

    for interface in _iter(_call(java_class, "getInterfaces", default=[])):
        names.extend(_java_class_names(interface, seen))
    names.extend(_java_class_names(_call(java_class, "getSuperclass", default=None), seen))
    return names


def _is_default(data_type: Any) -> bool:
    return _has_class(data_type, "DefaultDataType") or _type_name(data_type).startswith("undefined")


def _is_void(data_type: Any) -> bool:
    return _has_class(data_type, "VoidDataType") or (
        str(_call(data_type, "getName", default="")).lower() == "void"
        and int(_call(data_type, "getLength", default=0) or 0) == 0
    )


def _is_typedef(data_type: Any) -> bool:
    return _has_class(data_type, "TypeDef") or _has_class(data_type, "Typedef")


def _is_pointer(data_type: Any) -> bool:
    return _has_class(data_type, "Pointer") and hasattr(data_type, "getDataType")


def _is_array(data_type: Any) -> bool:
    return _has_class(data_type, "Array") and hasattr(data_type, "getNumElements")


def _is_structure(data_type: Any) -> bool:
    return _has_class(data_type, "Structure") and hasattr(data_type, "getComponents")


def _is_union(data_type: Any) -> bool:
    return _has_class(data_type, "Union") and hasattr(data_type, "getComponents")


def _is_enum(data_type: Any) -> bool:
    return _has_class(data_type, "Enum") and hasattr(data_type, "getNames")


def _is_bitfield(data_type: Any) -> bool:
    return _has_class(data_type, "BitFieldDataType") or (
        hasattr(data_type, "getBaseDataType")
        and (hasattr(data_type, "getDeclaredBitSize") or hasattr(data_type, "getBitSize"))
    )


def _is_function_definition(data_type: Any) -> bool:
    return _has_class(data_type, "FunctionDefinition") or (
        hasattr(data_type, "getArguments") and hasattr(data_type, "getReturnType")
    )


def _is_function_symbol(symbol: Any) -> bool:
    symbol_type = _call(symbol, "getSymbolType", default=None)
    return str(symbol_type).split(".")[-1].upper() == "FUNCTION"


def _type_name(data_type: Any) -> str:
    name = _call(data_type, "getName", default=None)
    if name is None or str(name) == "" or str(name).startswith("undefined"):
        category = _call(_call(data_type, "getCategoryPath", default=None), "getPath", default="root")
        path = _call(data_type, "getPathName", default=str(id(data_type)))
        name = f"unnamed_{abs(hash(f'{category}:{path}')):x}"
    return str(name)


def _base_kind(data_type: Any) -> str:
    lower = _type_name(data_type).lower()
    if _is_void(data_type):
        return "void"
    if _has_class(data_type, "BooleanDataType") or "bool" in lower:
        return "bool"
    if _has_class(data_type, "CharDataType") or lower == "char" or lower.endswith(" char"):
        return "char"
    if _has_class(data_type, "FloatDataType") or "float" in lower or "double" in lower:
        return "float"
    return "int"


def _is_signed_base(data_type: Any) -> bool:
    lower = _type_name(data_type).lower()
    if "unsigned" in lower or lower.startswith("u") or lower.startswith("uint") or "byte" in lower:
        return False
    kind = _base_kind(data_type)
    return kind not in ("bool", "void")


def _first_int(target: Any, fallback: int, *method_names: str) -> int:
    for method_name in method_names:
        value = _call(target, method_name, default=None)
        if isinstance(value, int) and value >= 0:
            return value
    return fallback


def _address_offset(address: Any) -> int:
    offset = _call(address, "getOffset", default=None)
    return int(address if offset is None else offset)


def _ghidra_version() -> str:
    try:
        import jpype  # type: ignore[import-not-found]

        if jpype.isJVMStarted():
            system = jpype.JClass("java.lang.System")
            return str(system.getProperty("application.version", "unknown"))
    except Exception:
        pass
    return "unknown"
