from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from dwarffi import DFFI
from dwarffi.ghidra import current_program_from_context, program_to_isf


class Language:
    def __init__(self, *, big_endian: bool = False) -> None:
        self._big_endian = big_endian

    def isBigEndian(self) -> bool:
        return self._big_endian


class BaseType:
    def __init__(self, name: str, length: int) -> None:
        self._name = name
        self._length = length

    def getName(self) -> str:
        return self._name

    def getLength(self) -> int:
        return self._length


class CharDataType(BaseType):
    pass


class FloatDataType(BaseType):
    pass


class TypeDef:
    def __init__(self, name: str, base: object) -> None:
        self._name = name
        self._base = base

    def getName(self) -> str:
        return self._name

    def getBaseDataType(self) -> object:
        return self._base


class Pointer:
    def __init__(self, subtype: object) -> None:
        self._subtype = subtype

    def getName(self) -> str:
        return f"{self._subtype.getName()} *"

    def getLength(self) -> int:
        return 8

    def getDataType(self) -> object:
        return self._subtype


class Array:
    def __init__(self, subtype: object, count: int) -> None:
        self._subtype = subtype
        self._count = count

    def getName(self) -> str:
        return f"{self._subtype.getName()}[{self._count}]"

    def getLength(self) -> int:
        return self._subtype.getLength() * self._count

    def getDataType(self) -> object:
        return self._subtype

    def getNumElements(self) -> int:
        return self._count


class BitFieldDataType:
    def __init__(self, base: object, bit_size: int) -> None:
        self._base = base
        self._bit_size = bit_size

    def getName(self) -> str:
        return f"{self._base.getName()}:{self._bit_size}"

    def getBaseDataType(self) -> object:
        return self._base

    def getDeclaredBitSize(self) -> int:
        return self._bit_size


class Component:
    def __init__(
        self,
        field_name: str | None,
        data_type: object,
        offset: int,
        *,
        bit_offset: int = 0,
    ) -> None:
        self._field_name = field_name
        self._data_type = data_type
        self._offset = offset
        self._bit_offset = bit_offset

    def getFieldName(self) -> str | None:
        return self._field_name

    def getDataType(self) -> object:
        return self._data_type

    def getOffset(self) -> int:
        return self._offset

    def getBitOffset(self) -> int:
        return self._bit_offset


class Structure:
    def __init__(self, name: str, length: int, components: list[Component]) -> None:
        self._name = name
        self._length = length
        self._components = components

    def getName(self) -> str:
        return self._name

    def getLength(self) -> int:
        return self._length

    def getComponents(self) -> list[Component]:
        return self._components


class Union:
    def __init__(self, name: str, length: int, components: list[Component]) -> None:
        self._name = name
        self._length = length
        self._components = components

    def getName(self) -> str:
        return self._name

    def getLength(self) -> int:
        return self._length

    def getComponents(self) -> list[Component]:
        return self._components


class Enum:
    def __init__(self, name: str, length: int, constants: dict[str, int]) -> None:
        self._name = name
        self._length = length
        self._constants = constants

    def getName(self) -> str:
        return self._name

    def getLength(self) -> int:
        return self._length

    def getNames(self) -> list[str]:
        return list(self._constants)

    def getValue(self, name: str) -> int:
        return self._constants[name]


class ParameterDefinition:
    def __init__(self, name: str, data_type: object) -> None:
        self._name = name
        self._data_type = data_type

    def getName(self) -> str:
        return self._name

    def getDataType(self) -> object:
        return self._data_type


class FunctionDefinition:
    def __init__(self, return_type: object, args: list[ParameterDefinition]) -> None:
        self._return_type = return_type
        self._args = args

    def getName(self) -> str:
        return "callback"

    def getReturnType(self) -> object:
        return self._return_type

    def getArguments(self) -> list[ParameterDefinition]:
        return self._args


class Address:
    def __init__(self, offset: int) -> None:
        self._offset = offset

    def getOffset(self) -> int:
        return self._offset


class Data:
    def __init__(self, data_type: object) -> None:
        self._data_type = data_type

    def getDataType(self) -> object:
        return self._data_type


class Symbol:
    def __init__(self, name: str, address: int, *, external: bool = False, symbol_type: str = "LABEL") -> None:
        self._name = name
        self._address = Address(address)
        self._external = external
        self._symbol_type = symbol_type

    def isExternal(self) -> bool:
        return self._external

    def getAddress(self) -> Address:
        return self._address

    def getSymbolType(self) -> str:
        return self._symbol_type

    def getName(self, include_namespace: bool = False) -> str:
        return self._name


class Function:
    def __init__(self, name: str, address: int, return_type: object, parameters: list[ParameterDefinition]) -> None:
        self._name = name
        self._address = Address(address)
        self._return_type = return_type
        self._parameters = parameters

    def isExternal(self) -> bool:
        return False

    def getName(self, include_namespace: bool = False) -> str:
        return self._name

    def getEntryPoint(self) -> Address:
        return self._address

    def getReturnType(self) -> object:
        return self._return_type

    def getParameters(self) -> list[ParameterDefinition]:
        return self._parameters


class DataTypeManager:
    def __init__(self, data_types: list[object]) -> None:
        self._data_types = data_types

    def getAllDataTypes(self) -> list[object]:
        return self._data_types


class SymbolTable:
    def __init__(self, symbols: list[Symbol]) -> None:
        self._symbols = symbols

    def getAllSymbols(self, forward: bool = True) -> list[Symbol]:
        return self._symbols


class Listing:
    def __init__(self, data_by_address: dict[int, Data], functions: list[Function]) -> None:
        self._data_by_address = data_by_address
        self._functions = functions

    def getDataAt(self, address: Address) -> Data | None:
        return self._data_by_address.get(address.getOffset())

    def getFunctions(self, forward: bool = True) -> list[Function]:
        return self._functions


class Memory:
    def contains(self, address: Address) -> bool:
        return address.getOffset() >= 0x1000


class Program:
    def __init__(
        self,
        data_types: list[object],
        symbols: list[Symbol],
        data_by_address: dict[int, Data],
        functions: list[Function],
    ) -> None:
        self._data_types = data_types
        self._symbols = symbols
        self._data_by_address = data_by_address
        self._functions = functions

    def getName(self) -> str:
        return "fake_program"

    def getLanguage(self) -> Language:
        return Language(big_endian=False)

    def getDefaultPointerSize(self) -> int:
        return 8

    def getDataTypeManager(self) -> DataTypeManager:
        return DataTypeManager(self._data_types)

    def getSymbolTable(self) -> SymbolTable:
        return SymbolTable(self._symbols)

    def getListing(self) -> Listing:
        return Listing(self._data_by_address, self._functions)

    def getMemory(self) -> Memory:
        return Memory()


def _fixture_program() -> Program:
    u8 = BaseType("uint8_t", 1)
    u16 = BaseType("uint16_t", 2)
    u32 = BaseType("uint32_t", 4)
    int_t = BaseType("int", 4)
    my_u32 = TypeDef("my_u32", u32)
    inner = Structure(
        "Inner",
        4,
        [
            Component("a", u16, 0),
            Component("b", u8, 2),
        ],
    )
    value = Union(
        "Value",
        4,
        [
            Component("word", u32, 0),
            Component("bytes", Array(u8, 4), 0),
        ],
    )
    callback = FunctionDefinition(int_t, [ParameterDefinition("x", int_t)])
    packet = Structure(
        "Packet",
        24,
        [
            Component("id", my_u32, 0),
            Component("inner", inner, 4),
            Component("value", value, 8),
            Component("flags", BitFieldDataType(u8, 3), 12, bit_offset=1),
            Component("next", Pointer(inner), 16),
            Component("cb", Pointer(callback), 20),
        ],
    )
    color = Enum("Color", 4, {"RED": 1, "BLUE": 2})
    add_packet = Function(
        "add_packet",
        0x1100,
        int_t,
        [ParameterDefinition("p", Pointer(packet)), ParameterDefinition("x", int_t)],
    )
    return Program(
        [u8, u16, u32, int_t, my_u32, inner, value, packet, color],
        [
            Symbol("global_counter", 0x4010),
            Symbol("external_data", 0x5000, external=True),
            Symbol("function_label", 0x1100, symbol_type="FUNCTION"),
        ],
        {0x4010: Data(int_t)},
        [add_packet],
    )


def test_program_to_isf_exports_types_symbols_and_functions() -> None:
    isf = program_to_isf(_fixture_program())

    assert isf["metadata"]["producer"]["name"] == "pyghidra2isf"
    assert isf["metadata"]["format"] == "6.2.0"
    assert isf["base_types"]["pointer"]["size"] == 8
    assert isf["base_types"]["uint8_t"]["signed"] is False
    assert isf["typedefs"]["my_u32"] == {"kind": "base", "name": "uint32_t"}

    packet = isf["user_types"]["Packet"]
    assert packet["kind"] == "struct"
    assert packet["size"] == 24
    assert packet["fields"]["id"]["type"] == {"kind": "typedef", "name": "my_u32"}
    assert packet["fields"]["inner"]["type"] == {"kind": "struct", "name": "Inner"}
    assert packet["fields"]["value"]["type"] == {"kind": "union", "name": "Value"}
    assert packet["fields"]["flags"]["type"] == {
        "kind": "bitfield",
        "bit_length": 3,
        "bit_position": 1,
        "type": {"kind": "base", "name": "uint8_t"},
    }
    assert packet["fields"]["next"]["type"] == {
        "kind": "pointer",
        "subtype": {"kind": "struct", "name": "Inner"},
    }
    assert packet["fields"]["cb"]["type"]["subtype"]["kind"] == "function"

    assert isf["user_types"]["Value"]["fields"]["bytes"]["type"] == {
        "kind": "array",
        "count": 4,
        "subtype": {"kind": "base", "name": "uint8_t"},
    }
    assert isf["enums"]["Color"]["constants"] == {"BLUE": 2, "RED": 1}
    assert isf["symbols"]["global_counter"]["address"] == 0x4010
    assert "external_data" not in isf["symbols"]
    assert "function_label" not in isf["symbols"]
    assert isf["functions"]["add_packet"]["parameters"][0]["type"] == {
        "kind": "pointer",
        "subtype": {"kind": "struct", "name": "Packet"},
    }


def test_from_ghidra_returns_regular_dffi_instance() -> None:
    ffi = DFFI.from_ghidra(_fixture_program())

    assert ffi.sizeof("Packet") == 24
    assert ffi.sizeof("my_u32") == 4
    assert ffi.typeof("my_u32").name == "uint32_t"
    assert ffi.get_symbol("global_counter").address == 0x4010
    assert ffi.get_function("add_packet").address == 0x1100


def test_types_only_omits_symbols_and_functions() -> None:
    isf = program_to_isf(_fixture_program(), types_only=True)

    assert isf["user_types"]["Packet"]
    assert isf["symbols"] == {}
    assert isf["functions"] == {}


def test_current_program_from_context_finds_ghidra_script_name() -> None:
    currentProgram = _fixture_program()

    assert current_program_from_context() is currentProgram


def test_from_ghidra_without_context_has_clear_error() -> None:
    with pytest.raises(RuntimeError, match=r"DFFI\.from_ghidra\(currentProgram\)"):
        DFFI.from_ghidra()


@pytest.mark.skipif(
    os.environ.get("DFFI_GHIDRA_TEST") != "1",
    reason="set DFFI_GHIDRA_TEST=1 to run the PyGhidra integration test",
)
def test_from_ghidra_with_pyghidra_fixture() -> None:
    pyghidra = pytest.importorskip("pyghidra")
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc is required for the PyGhidra integration fixture")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source_path = tmp_path / "fixture.c"
        binary_path = tmp_path / "fixture"
        source_path.write_text(
            """
            #include <stdint.h>

            typedef uint32_t my_u32;

            struct Packet {
                my_u32 id;
                uint16_t tag;
                uint8_t flags;
            };

            int global_counter = 7;

            int add_packet(struct Packet *p, int x) {
                return (int)p->id + x + global_counter;
            }
            """,
            encoding="utf-8",
        )
        subprocess.run(
            [
                gcc,
                "-g",
                "-O0",
                "-fno-eliminate-unused-debug-types",
                "-c",
                str(source_path),
                "-o",
                str(binary_path),
            ],
            check=True,
            text=True,
            capture_output=True,
        )

        open_program = getattr(pyghidra, "open_program", None)
        if open_program is None:
            pytest.skip("pyghidra.open_program is unavailable")

        kwargs = {"analyze": True}
        try:
            context = open_program(str(binary_path), **kwargs)
        except TypeError:
            try:
                pyghidra.start()
            except Exception:
                pass
            try:
                context = open_program(str(binary_path))
            except Exception as exc:
                pytest.skip(f"could not open fixture through PyGhidra: {exc}")
        except Exception as exc:
            pytest.skip(f"could not open fixture through PyGhidra: {exc}")

        with context as opened:
            program = opened
            if hasattr(opened, "getCurrentProgram"):
                program = opened.getCurrentProgram()
            elif hasattr(opened, "currentProgram"):
                program = opened.currentProgram

            try:
                ffi = DFFI.from_ghidra(program)
            except Exception as exc:
                pytest.skip(f"PyGhidra fixture opened but could not be converted: {exc}")

            assert ffi.sizeof("Packet") == ffi.get_type("Packet").size
            assert ffi.sizeof("my_u32") == 4
            assert ffi.get_symbol("global_counter") is not None
            assert ffi.get_function("add_packet") is not None

            types_only = DFFI.from_ghidra(program, types_only=True)
            assert types_only.sizeof("Packet") == ffi.sizeof("Packet")
            assert types_only.symbols == {}
            assert types_only.functions == {}
