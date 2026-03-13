from typing import TYPE_CHECKING, Any, Dict, Optional, Protocol, Tuple, Union, runtime_checkable

from .backend import LiveMemoryProxy
from .types import VtypeBaseType, VtypeEnum, VtypeStructField, VtypeUserType

if TYPE_CHECKING:
    from .instances import BoundArrayView, BoundTypeInstance, Ptr

VTYPE_CLASSES = (VtypeBaseType, VtypeEnum, VtypeUserType)
Vtype = Union[VtypeBaseType, VtypeEnum, VtypeUserType]

# Must be strings to prevent circular imports!
BoundType = Union['BoundTypeInstance', 'BoundArrayView', 'Ptr']

TypeInfoDict = Dict[str, Any]
MemoryBuffer = Union[bytes, bytearray, memoryview, LiveMemoryProxy]

@runtime_checkable
class StructLike(Protocol):
    """Protocol for struct.Struct and _FallbackIntStruct duck-types."""
    size: int
    format: str
    def unpack_from(self, buffer: Any, offset: int = 0) -> Tuple[Any, ...]: ...
    def pack_into(self, buffer: Any, offset: int, *args: Any) -> None: ...
    def unpack(self, buffer: Any) -> Tuple[Any, ...]: ...
    def pack(self, *args: Any) -> bytes: ...

@runtime_checkable
class TypeAccessor(Protocol):
    def get_type_size(self, type_info: TypeInfoDict) -> Optional[int]: ...
    def get_base_type(self, name: str) -> Optional[VtypeBaseType]: ...
    def get_enum(self, name: str) -> Optional[VtypeEnum]: ...
    def get_user_type(self, name: str) -> Optional[VtypeUserType]: ...
    def get_type(self, name: str) -> Optional[Vtype]: ...
    def _resolve_type_info(self, type_info: TypeInfoDict) -> TypeInfoDict: ...

FlatFieldTuple = Tuple[VtypeStructField, int, TypeInfoDict, Optional[Vtype]]
FlatFieldsDict = Dict[str, FlatFieldTuple]