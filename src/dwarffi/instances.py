import difflib
import struct
from typing import Any, Dict, Iterator, Optional, Tuple, Union

from .backend import LiveMemoryProxy
from .types import VtypeBaseType, VtypeEnum, VtypeUserType


def _wrap_integer(value: int, size_bytes: int, signed: bool) -> int:
    """
    Truncates and wraps an arbitrary Python integer into the boundaries of a
    C-style integer of `size_bytes` bytes, applying sign extension if needed.

    Args:
        value: The Python integer to truncate/wrap.
        size_bytes: The size of the C-type in bytes.
        signed: True if the type is signed.

    Returns:
        A wrapped integer within the valid range of the C-type.
    """
    if size_bytes == 0:
        return 0
    bits = size_bytes * 8
    mask = (1 << bits) - 1

    # 1. Truncate to the specified bit width
    val = value & mask

    # 2. If it's a signed type, check the sign bit and extend if necessary
    if signed:
        sign_bit = 1 << (bits - 1)
        if val & sign_bit:
            val -= 1 << bits

    return val


def _get_enum_struct(enum_def: VtypeEnum, vtype_accessor: Any) -> Tuple[struct.Struct, bool]:
    """
    Helper to reliably get a struct.Struct and signedness for an enum,
    falling back to synthesizing one from the enum's size if the base type is incomplete.
    """
    base_type_def = None
    if enum_def.base:
        base_type_def = vtype_accessor.get_base_type(enum_def.base)
        if base_type_def is None:
            raise KeyError(f"Underlying base type '{enum_def.base}' for enum '{enum_def.name}' not found.")

    # Auto-detect signedness from constants if possible
    has_negative = any(isinstance(v, int) and v < 0 for v in enum_def.constants.values())
    
    signed = has_negative
    if base_type_def is not None:
        signed = signed or bool(base_type_def.signed)
        
    size = enum_def.size if enum_def.size else (base_type_def.size if base_type_def else 4)
    if size == 0: 
        size = 4
        
    if size in (1, 2, 4, 8):
        fmt = {1: 'b', 2: 'h', 4: 'i', 8: 'q'}[size]
        if not signed:
            fmt = fmt.upper()
        endian = "<"
        if base_type_def is not None and getattr(base_type_def, "endian", "little") == "big":
            endian = ">"
        return struct.Struct(endian + fmt), signed
    
    # fallback to base type's compiled struct if odd size
    if base_type_def is not None:
        obj = base_type_def.get_compiled_struct()
        if obj is not None:
            return obj, signed
            
    raise ValueError(f"Cannot get compiled struct for enum '{enum_def.name}' (size={size})")


class BoundArrayView:
    """
    A high-performance view into an array field of a BoundTypeInstance.

    Provides Pythonic access (indexing, slicing, iteration) to C-style arrays 
    while mapping reads and writes directly to the parent's memory buffer.
    """

    __slots__ = (
        "_parent_instance",
        "_array_field_name",
        "_array_subtype_info",
        "_array_count",
        "_element_size",
        "_array_start_offset_in_parent",
        "_array_resolved_info",
        "_array_resolved_obj",
    )

    def __init__(
        self,
        parent_instance: "BoundTypeInstance",
        array_field_name: str,
        array_type_info: Dict[str, Any],
        array_start_offset_in_parent: int,
    ):
        """
        Internal constructor for creating an array view.
        """
        self._parent_instance = parent_instance
        self._array_field_name = array_field_name  # For error messages
        self._array_subtype_info = array_type_info.get("subtype")
        if self._array_subtype_info is None:
            raise ValueError(
                f"Array field '{array_field_name}' has no subtype information. type_info={array_type_info}"
            )
        self._array_count = array_type_info.get("count", 0)

        # Pre-calculate sizes for fast indexing
        self._element_size = parent_instance._instance_vtype_accessor.get_type_size(
            self._array_subtype_info
        )
        if self._element_size is None:
            raise ValueError(f"Cannot determine element size for array '{array_field_name}'.")

        self._array_start_offset_in_parent = array_start_offset_in_parent

        self._array_resolved_info = parent_instance._instance_vtype_accessor._resolve_type_info(self._array_subtype_info)
        kind = self._array_resolved_info.get("kind")
        t_name = self._array_resolved_info.get("name")
        self._array_resolved_obj = None
        if kind == "base" and t_name:
            self._array_resolved_obj = parent_instance._instance_vtype_accessor.get_base_type(t_name)
        elif kind == "enum" and t_name:
            self._array_resolved_obj = parent_instance._instance_vtype_accessor.get_enum(t_name)
        elif kind in ("struct", "union") and t_name:
            self._array_resolved_obj = parent_instance._instance_vtype_accessor.get_user_type(t_name)

    def __bytes__(self) -> bytes:
        """
        Returns an immutable byte snapshot of the entire array.
        """
        buf = self._parent_instance._instance_buffer
        start = self._parent_instance._instance_offset + self._array_start_offset_in_parent
        size = self._array_count * self._element_size
        return bytes(buf[start : start + size])

    def __getitem__(self, index: Union[int, slice]) -> Any:
        """
        Retrieves an element or a slice of elements from the array.
        
        Indices are bounds-checked. Complex subtypes (structs/unions) are returned 
        as BoundTypeInstances sharing the same memory buffer.
        """
        # Fast path for standard integer indexing
        if type(index) is int:
            if index < 0 or index >= self._array_count:
                raise IndexError(f"Index {index} out of bounds for array of size {self._array_count}")
            element_offset = self._array_start_offset_in_parent + (index * self._element_size)
            return self._parent_instance._read_data(
                self._array_resolved_info, self._array_resolved_obj, element_offset, f"{self._array_field_name}[{index}]"
            )

        if isinstance(index, slice):
            start, stop, step = index.indices(self._array_count)
            return [self[i] for i in range(start, stop, step)]

        raise TypeError(f"Array indices must be integers or slices, not {type(index).__name__}")

    def __setitem__(self, index: int, value: Any) -> None:
        """
        Writes a value to a specific array index.
        
        Clears the parent instance's cache for this field to ensure consistency.
        """
        if type(index) is int:
            if index < 0 or index >= self._array_count:
                raise IndexError(f"Index {index} out of bounds for array of size {self._array_count}")
                
            element_offset = self._array_start_offset_in_parent + (index * self._element_size)
            self._parent_instance._write_data(
                self._array_resolved_info, self._array_resolved_obj, element_offset, value, f"{self._array_field_name}[{index}]"
            )
            try:
                del self._parent_instance._instance_cache[self._array_field_name]
            except KeyError:
                pass
            return
        raise TypeError("Array assignment requires an integer index.")

    def __len__(self) -> int:
        """Returns the number of elements in the array."""
        return self._array_count

    def __iter__(self) -> Iterator[Any]:
        """Iterates over the elements of the array."""
        for i in range(self._array_count):
            yield self[i]

    def __repr__(self) -> str:
        preview_count = min(self._array_count, 3)
        items_preview = [repr(self[i]) for i in range(preview_count)]
        if self._array_count > preview_count:
            items_preview.append("...")
        return f"<BoundArrayView Field='{self._array_field_name}' Count={self._array_count} Items=[{', '.join(items_preview)}]>"

    def __add__(self, offset: int) -> "Ptr":
        """
        Implements C-style pointer decay. 
        `arr + 5` returns a Ptr to the 5th element of the array.
        """
        if not isinstance(offset, int):
            return NotImplemented
            
        # Natively check for the global base address
        base_addr = getattr(self._parent_instance, "_base_address", None)
        if base_addr is None:
            base_addr = self._parent_instance._instance_offset
        else:
            base_addr = base_addr + self._parent_instance._instance_offset
            
        base_addr += self._array_start_offset_in_parent
        return Ptr(
            base_addr + (offset * self._element_size),
            self._array_subtype_info,
            self._parent_instance._instance_vtype_accessor,
        )

    def __eq__(self, other: Any) -> bool:
        """Compares array contents against Python lists or other Array Views."""
        if isinstance(other, (list, BoundArrayView)):
            if len(self) != len(other):
                return False
            return all(self[i] == other[i] for i in range(len(self)))
        return False

    def __ne__(self, other: Any) -> bool:
        return not self.__eq__(other)


class BoundTypeInstance:
    """
    An instance of a C-type bound to a memory buffer.

    This is the core object used for memory interaction. For structs and unions, 
    fields are accessed via attribute lookup. For base types and enums, values 
    can be accessed via indexing `inst[0]` or numeric casting `int(inst)`.
    """
    __slots__ = ("_instance_type_name", 
                 "_instance_type_def", 
                 "_instance_buffer", 
                 "_instance_vtype_accessor", 
                 "_instance_offset", 
                 "_instance_cache", 
                 "_flat_fields",
                 "_instance_unpack_struct", 
                 "_instance_pack_struct",
                 "_base_address")

    def __init__(
        self,
        type_name: str,
        type_def: Union[VtypeUserType, VtypeBaseType, VtypeEnum],
        buffer: Union[bytearray, memoryview, LiveMemoryProxy],
        vtype_accessor: Any,
        instance_offset_in_buffer: int = 0,
        base_address: Optional[int] = None,
    ):
        # Determine if we have a proxy or a native buffer
        is_proxy = getattr(buffer, "backend", None) is not None
        
        if not is_proxy and not isinstance(buffer, (bytearray, memoryview, bytes)):
             raise TypeError("BoundTypeInstance expects a byte-like object or LiveMemoryProxy.")

        # Wrap native buffers in memoryview for zero-copy slicing/comparisons
        if not is_proxy and not isinstance(buffer, memoryview):
            buffer = memoryview(buffer)
        # Fast initialization bypassing __setattr__ completely to prevent recursion
        object.__setattr__(self, "_instance_type_name", type_name)
        object.__setattr__(self, "_instance_type_def", type_def)
        object.__setattr__(self, "_instance_buffer", buffer)
        object.__setattr__(self, "_instance_vtype_accessor", vtype_accessor)
        object.__setattr__(self, "_instance_offset", instance_offset_in_buffer)
        object.__setattr__(self, "_instance_cache", {})
        object.__setattr__(self, "_base_address", base_address)
        
        # Lazy loading: Initialize to None to defer lookup overhead
        object.__setattr__(self, "_flat_fields", None)

        if getattr(buffer, "backend", None) is not None:
            object.__setattr__(self, "_instance_unpack_struct", self._instance_unpack_proxy)
            object.__setattr__(self, "_instance_pack_struct", self._instance_pack_proxy)
        else:
            object.__setattr__(self, "_instance_unpack_struct", self._instance_unpack_native)
            object.__setattr__(self, "_instance_pack_struct", self._instance_pack_native)

    def _instance_unpack_native(self, compiled_struct_obj: Any, offset: int) -> Any:
        """Fast path: Zero-copy native buffer read."""
        return compiled_struct_obj.unpack_from(self._instance_buffer, offset)[0]

    def _instance_pack_native(self, compiled_struct_obj: Any, offset: int, value: Any) -> None:
        """Fast path: Zero-copy native buffer write."""
        compiled_struct_obj.pack_into(self._instance_buffer, offset, value)

    def _instance_unpack_proxy(self, compiled_struct_obj: Any, offset: int) -> Any:
        """Proxy path: Slice-based read for non-buffer-protocol objects."""
        data = self._instance_buffer[offset : offset + compiled_struct_obj.size]
        if hasattr(compiled_struct_obj, "unpack"):
            return compiled_struct_obj.unpack(data)[0]
        return compiled_struct_obj.unpack_from(data, 0)[0]

    def _instance_pack_proxy(self, compiled_struct_obj: Any, offset: int, value: Any) -> None:
        """Proxy path: Slice-based write for non-buffer-protocol objects."""
        if hasattr(compiled_struct_obj, "pack"):
            data = compiled_struct_obj.pack(value)
        else:
            data = bytearray(compiled_struct_obj.size)
            compiled_struct_obj.pack_into(data, 0, value)
        self._instance_buffer[offset : offset + compiled_struct_obj.size] = data

    def _get_value(self) -> Any:
        """Internal: Deserializes the value if this is a base type or enum."""
        if isinstance(self._instance_type_def, VtypeUserType):
            raise AttributeError(
                f"'{self._instance_type_name}' is a struct/union and does not have a direct value. Access its fields instead."
            )
        if isinstance(self._instance_type_def, VtypeBaseType):
            base_type_def = self._instance_type_def
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if base_type_def.size == 0:
                return None
            if compiled_struct_obj is None:
                raise ValueError(f"Cannot get compiled struct for base type '{base_type_def.name}'")
            try:
                return self._instance_unpack_struct(compiled_struct_obj, self._instance_offset)
            except struct.error as e:
                raise struct.error(
                    f"Error unpacking value for base type '{base_type_def.name}': {e}"
                ) from e
        elif isinstance(self._instance_type_def, VtypeEnum):
            enum_def = self._instance_type_def
            compiled_struct_obj, _ = _get_enum_struct(enum_def, self._instance_vtype_accessor)
            try:
                int_val = self._instance_unpack_struct(compiled_struct_obj, self._instance_offset)
                return EnumInstance(enum_def, int_val)
            except struct.error as e:
                raise struct.error(f"Error unpacking value for enum '{enum_def.name}': {e}") from e
        else:
            raise TypeError(
                f"Cannot get value on internal type: {type(self._instance_type_def).__name__}"
            )

    def _set_value(self, new_value: Any) -> None:
        """Internal: Serializes and writes a value to memory."""
        if isinstance(self._instance_type_def, VtypeUserType):
            raise AttributeError(
                f"Cannot set a direct value on a struct/union '{self._instance_type_name}'. Set its fields instead."
            )
        if isinstance(self._instance_type_def, VtypeBaseType):
            base_type_def = self._instance_type_def
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if base_type_def.size == 0:
                if new_value is not None:
                    raise ValueError("Cannot assign value to void type.")
                return
            if compiled_struct_obj is None:
                raise ValueError(
                    f"Cannot get compiled struct for base type '{base_type_def.name}' to write value."
                )

            if isinstance(new_value, int):
                new_value = _wrap_integer(new_value, base_type_def.size, bool(base_type_def.signed))

            try:
                self._instance_pack_struct(compiled_struct_obj, self._instance_offset, new_value)
            except struct.error as e:
                raise struct.error(
                    f"Error packing value for base type '{base_type_def.name}': {e}"
                ) from e
        elif isinstance(self._instance_type_def, VtypeEnum):
            enum_def = self._instance_type_def
            compiled_struct_obj, signed = _get_enum_struct(enum_def, self._instance_vtype_accessor)

            int_val_to_write: int
            if isinstance(new_value, EnumInstance):
                int_val_to_write = new_value._value
            elif isinstance(new_value, int):
                int_val_to_write = new_value
            elif isinstance(new_value, str):
                found_val = enum_def.constants.get(new_value)
                if found_val is None:
                    raise ValueError(
                        f"Enum constant name '{new_value}' not found in enum '{enum_def.name}'."
                    )
                int_val_to_write = found_val
            else:
                raise TypeError(
                    f"Cannot write type '{type(new_value)}' to enum instance. Expected EnumInstance, int, or str."
                )
            if isinstance(int_val_to_write, int):
                int_val_to_write = _wrap_integer(
                    int_val_to_write, compiled_struct_obj.size, signed
                )
            try:
                self._instance_pack_struct(compiled_struct_obj, self._instance_offset, int_val_to_write)
            except struct.error as e:
                raise struct.error(f"Error packing value for enum '{enum_def.name}': {e}") from e
        else:
            raise TypeError(
                f"Setter not applicable to internal type: {type(self._instance_type_def).__name__}"
            )

    def __int__(self) -> int:
        if isinstance(self._instance_type_def, (VtypeBaseType, VtypeEnum)):
            return int(self._get_value())
        if isinstance(self._instance_type_def, VtypeUserType):
            raise TypeError(f"Cannot convert struct/union '{self._instance_type_name}' to int")
        return NotImplemented

    def __index__(self) -> int:
        return self.__int__()

    def __float__(self) -> float:
        if (
            isinstance(self._instance_type_def, VtypeBaseType)
            and self._instance_type_def.kind == "float"
        ):
            return float(self._get_value())
        raise TypeError(f"Cannot convert type '{self._instance_type_name}' to float")

    def __bool__(self) -> bool:
        if isinstance(self._instance_type_def, (VtypeBaseType, VtypeEnum)):
            val = self._get_value()
            if isinstance(val, EnumInstance):
                return bool(int(val))
            return bool(val)
        return True

    def __getitem__(self, index: int) -> Any:
        if index == 0 and isinstance(self._instance_type_def, (VtypeBaseType, VtypeEnum)):
            return self._get_value()
        if isinstance(self._instance_type_def, VtypeUserType):
            if index == 0:
                return self
            raise IndexError(
                "Struct/Union index out of range (only [0] is supported for dereferencing)"
            )
        raise TypeError(f"'{self._instance_type_name}' object is not subscriptable")

    def __setitem__(self, index: int, value: Any):
        if index == 0 and isinstance(self._instance_type_def, (VtypeBaseType, VtypeEnum)):
            self._set_value(value)
            return
        if isinstance(self._instance_type_def, VtypeUserType):
            raise TypeError(
                "Cannot overwrite entire struct/union via subscript assignment. Assign to fields instead."
            )
        raise TypeError(f"'{self._instance_type_name}' object does not support item assignment")

    def _read_data(
        self,
        resolved_info: Dict[str, Any],
        resolved_obj: Any,
        field_offset_in_struct: int,
        field_name_for_error: str,
    ) -> Any:
        kind = resolved_info["kind"]
        absolute_field_offset = self._instance_offset + field_offset_in_struct

        if kind == "base":
            base_type_def = resolved_obj # Use cached object
            if base_type_def is None:
                name = resolved_info.get("name", "unknown")
                raise KeyError(f"Required base type '{name}' for field '{field_name_for_error}' not found.")
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if base_type_def.size == 0:
                return None
            return self._instance_unpack_struct(compiled_struct_obj, absolute_field_offset)

        elif kind == "pointer":
            ptr_base_type = self._instance_vtype_accessor.get_base_type("pointer")
            if ptr_base_type is None:
                raise KeyError("Base type 'pointer' not defined in loaded ISF files. Cannot read pointer field.")
            compiled_struct_obj = ptr_base_type.get_compiled_struct()
            address = self._instance_unpack_struct(compiled_struct_obj, absolute_field_offset)
            return Ptr(address, resolved_info.get("subtype"), self._instance_vtype_accessor)

        elif kind == "array":
            return BoundArrayView(
                self, field_name_for_error, resolved_info, field_offset_in_struct
            )

        elif kind in ("struct", "union"):
            user_type_def = resolved_obj # Use cached object
            if user_type_def is None:
                name = resolved_info.get("name", "unknown")
                raise KeyError(f"Struct/Union definition '{name}' for field '{field_name_for_error}' not found.")
            return BoundTypeInstance(
                resolved_info.get("name", "anonymous"),
                user_type_def,
                self._instance_buffer,
                self._instance_vtype_accessor,
                absolute_field_offset,
                self._base_address
            )

        elif kind == "enum":
            enum_def = resolved_obj # Use cached object
            compiled_struct_obj, _ = _get_enum_struct(enum_def, self._instance_vtype_accessor)
            int_val = self._instance_unpack_struct(compiled_struct_obj, absolute_field_offset)
            return EnumInstance(enum_def, int_val)

        elif kind == "bitfield":
            bit_length = resolved_info["bit_length"]
            bit_position = resolved_info["bit_position"]
            underlying_base_name = resolved_info["type"]["name"]
            underlying_base_def = self._instance_vtype_accessor.get_base_type(underlying_base_name)
            compiled_struct_obj = underlying_base_def.get_compiled_struct()
            
            # Prevent buffer overrun when structs are extremely packed
            size = underlying_base_def.size
            buf_slice = bytearray(self._instance_buffer[absolute_field_offset : absolute_field_offset + size])
            if len(buf_slice) < size:
                buf_slice.extend(b'\x00' * (size - len(buf_slice)))
            
            storage_unit_val = compiled_struct_obj.unpack_from(buf_slice, 0)[0]
            
            mask = (1 << bit_length) - 1
            val = (storage_unit_val >> bit_position) & mask
            
            # Sign extension for standard C-integer behavior
            if underlying_base_def.signed and (val & (1 << (bit_length - 1))):
                val -= (1 << bit_length)
                
            return val

        elif kind == "function":
            return f"<FunctionType: {resolved_info.get('name', 'anon_func')}>"
        elif kind == "void":
            return None
        else:
            raise ValueError(
                f"Unsupported/invalid type kind '{kind}' for field '{field_name_for_error}'."
            )

    def _write_data(
        self,
        resolved_info: Dict[str, Any],
        resolved_obj: Any,
        field_offset_in_struct: int,
        value_to_write: Any,
        field_name_for_error: str,
    ):
        kind = resolved_info["kind"]
        absolute_field_offset = self._instance_offset + field_offset_in_struct

        if kind == "base":
            base_type_def = resolved_obj
            if base_type_def is None:
                name = resolved_info.get("name", "unknown")
                raise KeyError(f"Required base type '{name}' for field '{field_name_for_error}' not found.")
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if base_type_def.size == 0:
                return
            if (
                base_type_def.signed is False
                and isinstance(value_to_write, int)
                and value_to_write < 0
            ):
                value_to_write = value_to_write % (1 << (base_type_def.size * 8))
            if base_type_def.signed is True and isinstance(value_to_write, int):
                bits = base_type_def.size * 8
                max_signed = (1 << (bits - 1)) - 1
                min_signed = -(1 << (bits - 1))
                if value_to_write > max_signed:
                    value_to_write = value_to_write - (1 << bits)
                if value_to_write < min_signed:
                    value_to_write = ((value_to_write + (1 << bits)) % (1 << bits)) + min_signed
            if isinstance(value_to_write, int):
                value_to_write = _wrap_integer(
                    value_to_write, base_type_def.size, bool(base_type_def.signed)
                )
            self._instance_pack_struct(compiled_struct_obj, absolute_field_offset, value_to_write)

        elif kind == "pointer":
            ptr_base_type = self._instance_vtype_accessor.get_base_type("pointer")
            if ptr_base_type is None:
                raise KeyError("Base type 'pointer' not defined in loaded ISF files. Cannot write pointer field.")
            compiled_struct_obj = ptr_base_type.get_compiled_struct()
            address_to_write = (
                value_to_write.address if isinstance(value_to_write, Ptr) else value_to_write
            )
            if isinstance(address_to_write, int):
                address_to_write = _wrap_integer(address_to_write, ptr_base_type.size, False)
            self._instance_pack_struct(compiled_struct_obj, absolute_field_offset, address_to_write)

        elif kind == "enum":
            enum_def = resolved_obj
            compiled_struct_obj, signed = _get_enum_struct(enum_def, self._instance_vtype_accessor)

            if isinstance(value_to_write, EnumInstance):
                int_val_to_write = value_to_write._value
            elif isinstance(value_to_write, int):
                int_val_to_write = value_to_write
            elif isinstance(value_to_write, str):
                int_val_to_write = enum_def.constants.get(value_to_write)
                if int_val_to_write is None:
                    raise ValueError(f"Enum constant '{value_to_write}' not found.")
            else:
                raise TypeError("Expected int, str, or EnumInstance for enum assignment.")

            if isinstance(int_val_to_write, int):
                int_val_to_write = _wrap_integer(
                    int_val_to_write, compiled_struct_obj.size, signed
                )
            self._instance_pack_struct(compiled_struct_obj, absolute_field_offset, int_val_to_write)

        elif kind == "bitfield":
            bit_length = resolved_info["bit_length"]
            bit_position = resolved_info["bit_position"]
            underlying_base_name = resolved_info["type"]["name"]
            underlying_base_def = self._instance_vtype_accessor.get_base_type(underlying_base_name)
            compiled_struct_obj = underlying_base_def.get_compiled_struct()
            
            # Prevent buffer overrun when structs are extremely packed
            size = underlying_base_def.size
            buf_slice = bytearray(self._instance_buffer[absolute_field_offset : absolute_field_offset + size])
            pad_len = size - len(buf_slice)
            if pad_len > 0:
                buf_slice.extend(b'\x00' * pad_len)
            
            current_storage_val = compiled_struct_obj.unpack_from(buf_slice, 0)[0]
            mask = (1 << bit_length) - 1
            value_to_set = value_to_write & mask
            new_storage_val = (current_storage_val & ~(mask << bit_position)) | (
                value_to_set << bit_position
            )
            compiled_struct_obj.pack_into(buf_slice, 0, new_storage_val)
            
            # Write back up to valid buffer size limit
            valid_len = size - pad_len
            self._instance_buffer[absolute_field_offset : absolute_field_offset + valid_len] = buf_slice[:valid_len]

        elif kind in ("array", "struct", "union"):
            raise NotImplementedError(
                f"Direct assignment to field '{field_name_for_error}' of type '{kind}' is not supported."
            )

    def _find_field(self, name: str):
        """Recursively checks for anonymous fields."""
        if not isinstance(self._instance_type_def, VtypeUserType):
            return None, None

        def _search(type_def, target_name, current_offset):
            if not isinstance(type_def, VtypeUserType):
                return None, None
            if target_name in type_def.fields:
                f_def = type_def.fields[target_name]
                return f_def, current_offset + f_def.offset

            for _, f_def in type_def.fields.items():
                if f_def.anonymous:
                    sub_type = self._instance_vtype_accessor.get_user_type(
                        f_def.type_info.get("name")
                    )
                    if sub_type:
                        res, off = _search(sub_type, target_name, current_offset + f_def.offset)
                        if res:
                            return res, off
            return None, None

        return _search(self._instance_type_def, name, 0)

    def __getattr__(self, name: str) -> Any:
        """
        Handles field access for structs and unions.
        
        Leverages O(1) flattened field lookups to support anonymous nested structs
        without recursive overhead.
        """
        # Block recursion and handle python dunders gracefully
        if name[0] == '_' and (name.startswith('_instance_') or name.startswith('__')):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

        # Lazy Evaluation Core: Only resolve flat_fields on the FIRST actual field access
        flat_fields = self._flat_fields
        if flat_fields is None:
            if type(self._instance_type_def) is VtypeUserType:
                flat_fields = self._instance_type_def.get_flattened_fields(self._instance_vtype_accessor)
            else:
                # Use False as a fast sentinel value so we know it's not a struct,
                # while an empty struct dictionary {} remains distinguishable.
                flat_fields = False
            object.__setattr__(self, "_flat_fields", flat_fields)

        # Check 'is not False' because an empty struct dict '{}' evaluates to False in python
        if flat_fields is not False:
            if name in self._instance_cache:
                return self._instance_cache[name]

            if name in flat_fields:
                _, field_offset, resolved_info, resolved_obj = flat_fields[name]
                val = self._read_data(resolved_info, resolved_obj, field_offset, name)
                
                if resolved_info["kind"] in ("struct", "union", "array"):
                    self._instance_cache[name] = val
                return val

            error_msg = f"'{self._instance_type_name}' has no attribute '{name}'"
            matches = difflib.get_close_matches(name, flat_fields.keys(), n=1, cutoff=0.6)
            if matches:
                error_msg += f". Did you mean '{matches[0]}'?"
                
            raise AttributeError(error_msg)

        raise AttributeError(
            f"Type '{self._instance_type_name}' has no attribute '{name}'. Use '[0]' or cast for base/enum types."
        )

    def __setattr__(self, name: str, new_value: Any) -> None:
        """
        Handles field writes for structs and unions.
        
        Updates the underlying buffer and invalidates relevant caches.
        """
        if name[0] == '_' and (name.startswith('_instance_') or name.startswith('__')):
            object.__setattr__(self, name, new_value)
            return

        # Lazy Evaluation Core
        flat_fields = self._flat_fields
        if flat_fields is None:
            if type(self._instance_type_def) is VtypeUserType:
                flat_fields = self._instance_type_def.get_flattened_fields(self._instance_vtype_accessor)
            else:
                flat_fields = False
            object.__setattr__(self, "_flat_fields", flat_fields)

        if flat_fields is not False:
            if name not in flat_fields:
                object.__setattr__(self, name, new_value)
                return
            
            _, field_offset, resolved_info, resolved_obj = flat_fields[name]
            
            if resolved_info["kind"] == "array":
                # Allow direct assignment only for byte/char arrays
                if isinstance(new_value, str):
                    data = new_value.encode("utf-8")
                elif isinstance(new_value, (bytes, bytearray, memoryview)):
                    data = bytes(new_value)
                else:
                    raise NotImplementedError(
                        f"Direct assignment to array field '{name}' is not supported."
                    )

                # Resolve element type and ensure it's 1 byte
                subtype_info = resolved_info.get("subtype")
                if subtype_info is None:
                    raise ValueError(f"Array field '{name}' missing subtype info.")

                elem_size = self._instance_vtype_accessor.get_type_size(subtype_info)
                if elem_size != 1:
                    raise NotImplementedError(
                        f"Direct assignment to non-byte array field '{name}' is not supported."
                    )

                count = resolved_info.get("count", 0)
                if count <= 0:
                    return

                # Compute array start in underlying buffer
                start = self._instance_offset + field_offset
                end = start + count


                payload = data[: max(0, count - 1)]
                full_data = payload.ljust(count, b"\x00")
                self._instance_buffer[start : end] = full_data

                # Invalidate cache for this field if present
                try:
                    del self._instance_cache[name]
                except KeyError:
                    pass
                return
            self._write_data(resolved_info, resolved_obj, field_offset, new_value, name)
            
            # Try/except is faster than checking 'in' for cache invalidation
            try:
                del self._instance_cache[name]
            except KeyError:
                pass
            return

        raise AttributeError(
            f"Cannot set attribute '{name}' on type '{self._instance_type_name}'. Use '[0]' for base/enum types."
        )

    # --- Numeric Magic Methods (For Base Types & Enums) ---
    def _assert_numeric(self):
        if isinstance(self._instance_type_def, VtypeUserType):
            raise TypeError(
                f"unsupported operand type(s) for struct/union '{self._instance_type_name}'"
            )

    def __add__(self, other):
        self._assert_numeric()
        return self._get_value() + other

    def __radd__(self, other):
        self._assert_numeric()
        return other + self._get_value()

    def __sub__(self, other):
        self._assert_numeric()
        return self._get_value() - other

    def __rsub__(self, other):
        self._assert_numeric()
        return other - self._get_value()

    def __mul__(self, other):
        self._assert_numeric()
        return self._get_value() * other

    def __rmul__(self, other):
        self._assert_numeric()
        return other * self._get_value()

    # --- Rich Comparisons ---
    def __lt__(self, other):
        if isinstance(self._instance_type_def, VtypeUserType):
            return NotImplemented
        return self._get_value() < other

    def __le__(self, other):
        if isinstance(self._instance_type_def, VtypeUserType):
            return NotImplemented
        return self._get_value() <= other

    def __gt__(self, other):
        if isinstance(self._instance_type_def, VtypeUserType):
            return NotImplemented
        return self._get_value() > other

    def __ge__(self, other):
        if isinstance(self._instance_type_def, VtypeUserType):
            return NotImplemented
        return self._get_value() >= other

    def __eq__(self, other):
        # 1. Compare against another BoundTypeInstance
        if isinstance(other, BoundTypeInstance):
            # Same memory reference
            if (
                self._instance_buffer is other._instance_buffer
                and self._instance_offset == other._instance_offset
            ):
                return True
            # Same type name and same exact byte values
            if self._instance_type_name == other._instance_type_name:
                size = self._instance_type_def.size
                v1 = self._instance_buffer[self._instance_offset : self._instance_offset + size]
                v2 = other._instance_buffer[other._instance_offset : other._instance_offset + size]
                return v1 == v2

            # If both are primitive/enum types, try comparing their actual values
            if not isinstance(self._instance_type_def, VtypeUserType) and not isinstance(
                other._instance_type_def, VtypeUserType
            ):
                return self._get_value() == other._get_value()
            return False

        # 2. Compare directly against raw bytes/bytearray/memoryview
        if isinstance(other, (bytes, bytearray, memoryview)):
            size = self._instance_type_def.size
            if len(other) != size:
                return False
            
            # Zero-copy comparison against the byte-like object
            v1 = self._instance_buffer[self._instance_offset : self._instance_offset + size]
            return v1 == other

        # 3. Compare against native Python types (int, float, str, etc.)
        if not isinstance(self._instance_type_def, VtypeUserType):
            return self._get_value() == other

        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bytes__(self) -> bytes:
        size = self._instance_type_def.size
        if size == 0:
            return b""
        start = self._instance_offset
        return bytes(self._instance_buffer[start : start + size])

    def __repr__(self) -> str:
        return f"<BoundTypeInstance Type='{self._instance_type_name}' Kind='{self._instance_type_def.__class__.__name__}' AtOffset={self._instance_offset}>"

    def __dir__(self):
        """Allows tab-completion to see recursive anonymous fields."""
        items = list(super().__dir__())
        if isinstance(self._instance_type_def, VtypeUserType):
            # Add direct fields
            items.extend(self._instance_type_def.fields.keys())
            
            # Recursively add fields from anonymous members
            def _get_anon_fields(t_def):
                extra = []
                for f in t_def.fields.values():
                    if f.anonymous:
                        sub_t = self._instance_vtype_accessor.get_type(f.type_info.get("name"))
                        if isinstance(sub_t, VtypeUserType):
                            extra.extend(sub_t.fields.keys())
                            extra.extend(_get_anon_fields(sub_t))
                return extra

            items.extend(_get_anon_fields(self._instance_type_def))
        
        # Filter duplicates and hide the internal cache from tab-completion
        return sorted(list(set(a for a in items if a != "_instance_cache")))

class Ptr:
    """
    Represents a C-style pointer (memory address + type context).

    Provides C-style pointer arithmetic. Adding an integer to a Ptr shifts 
    the address by `offset * sizeof(target_type)`.
    """
    __slots__ = "address", "_subtype_info", "_vtype_accessor"

    def __init__(self, address: int, subtype_info: Optional[Dict[str, Any]], vtype_accessor: Any):
        self.address = address
        self._subtype_info = subtype_info
        self._vtype_accessor = vtype_accessor


    def __repr__(self) -> str:
        return f"<Ptr to {self.points_to_type_name} at {hex(self.address)}>"

    @property
    def points_to_type_info(self) -> Optional[Dict[str, Any]]:
        """Returns the raw ISF type dictionary of the target."""
        return self._subtype_info

    @property
    def points_to_type_name(self) -> str:
        """Returns the string name of the pointed-to type."""
        if not self._subtype_info:
            return "void"

        # If it's a Vtype object, use the .name attribute
        if hasattr(self._subtype_info, "name"):
            return self._subtype_info.name

        # If it's a raw ISF dictionary, use .get()
        return self._subtype_info.get("name", "void")

    def deref(self) -> Any:
        """
        Dereferences the pointer. If it points to another pointer, it actively resolves 
        the pointer chain natively.
        Requires the DFFI engine to have a configured MemoryBackend.
        """
        if not self._subtype_info or self._subtype_info.get("name") == "void":
            raise TypeError("Cannot dereference a void pointer.")
            
        subtype = self._vtype_accessor._resolve_type_info(self._subtype_info)
        
        # If the target is ALSO a pointer, we must read the raw pointer value from memory
        if isinstance(subtype, dict) and subtype.get("kind") == "pointer":
            if not getattr(self._vtype_accessor, "backend", None):
                raise RuntimeError("Cannot dereference pointer chain without a configured memory backend.")
            ptr_base_type = self._vtype_accessor.get_base_type("pointer")
            size = ptr_base_type.size
            data = self._vtype_accessor.backend.read(self.address, size)
            target_addr = ptr_base_type.get_compiled_struct().unpack(data)[0]
            return Ptr(target_addr, subtype.get("subtype"), self._vtype_accessor)
            
        return self._vtype_accessor.from_address(subtype, self.address)

    def __getitem__(self, index: int) -> Any:
        """
        Allows C-style array indexing on pointers (e.g., ptr[0], ptr[5]).
        Equivalent to (ptr + index).deref()
        """
        if not isinstance(index, int):
            raise TypeError("Pointer indices must be integers.")
            
        target_ptr = self + index
        return target_ptr.deref()

    # --- Type Conversions ---
    def __int__(self) -> int:
        return self.address

    def __index__(self) -> int:
        return self.address

    def __bool__(self) -> bool:
        return self.address != 0

    def __hash__(self) -> int:
        return hash((self.address, self.points_to_type_name))

    # --- Pointer Arithmetic ---
    def __add__(self, offset: int) -> "Ptr":
        if not isinstance(offset, int):
            return NotImplemented
        size = self._vtype_accessor.get_type_size(self._subtype_info) or 1
        return Ptr(self.address + (offset * size), self._subtype_info, self._vtype_accessor)

    def __sub__(self, other: Union[int, "Ptr"]) -> Union["Ptr", int]:
        if isinstance(other, int):
            size = self._vtype_accessor.get_type_size(self._subtype_info) or 1
            return Ptr(self.address - (other * size), self._subtype_info, self._vtype_accessor)
        elif isinstance(other, Ptr):
            # C-style pointer subtraction yields the number of elements between them
            size = self._vtype_accessor.get_type_size(self._subtype_info) or 1
            return (self.address - other.address) // size
        return NotImplemented

    # --- Comparisons ---
    def __eq__(self, other):
        if isinstance(other, Ptr):
            return self.address == other.address
        if isinstance(other, int):
            return self.address == other
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self.address < (other.address if isinstance(other, Ptr) else other)

    def __le__(self, other):
        return self.address <= (other.address if isinstance(other, Ptr) else other)

    def __gt__(self, other):
        return self.address > (other.address if isinstance(other, Ptr) else other)

    def __ge__(self, other):
        return self.address >= (other.address if isinstance(other, Ptr) else other)


class EnumInstance:
    """
    A specific value of an enumeration.
    
    Supports string comparison against constant names and integer comparison 
    against the underlying value.
    """
    __slots__ = "_enum_def", "_value"

    def __init__(self, enum_def: VtypeEnum, value: int):
        self._enum_def = enum_def
        self._value = value

    @property
    def name(self) -> Optional[str]:
        """Returns the name of the enumeration constant for this value."""
        return self._enum_def.get_name_for_value(self._value)

    def __repr__(self) -> str:
        name_part = (
            f"{self._enum_def.name}.{self.name}" if self.name else f"{self._enum_def.name} (value)"
        )
        return f"<EnumInstance {name_part} ({self._value})>"

    def __int__(self) -> int:
        return self._value

    def __eq__(self, other):
        if isinstance(other, EnumInstance):
            return self._value == other._value and self._enum_def.name == other._enum_def.name
        if isinstance(other, int):
            return self._value == other
        if isinstance(other, str):
            return self.name == other
        return False
