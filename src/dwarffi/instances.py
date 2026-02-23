import struct
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

from .types import VtypeUserType, VtypeBaseType, VtypeEnum

# Prevents circular import with parser.py
if TYPE_CHECKING:
    from .parser import VtypeJson


def _wrap_integer(value: int, size_bytes: int, signed: bool) -> int:
    """
    Truncates and wraps an arbitrary Python integer into the boundaries of a 
    C-style integer of `size_bytes` bytes, applying sign extension if needed.
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
            val -= (1 << bits)
            
    return val


class BoundArrayView:
    """A view into an array field of a BoundTypeInstance, allowing get/set of elements."""
    __slots__ = '_parent_instance', '_array_field_name', '_array_subtype_info', '_array_count', '_element_size', '_array_start_offset_in_parent'

    def __init__(self, parent_instance: 'BoundTypeInstance', array_field_name: str,
                 array_type_info: Dict[str, Any], array_start_offset_in_parent: int):
        self._parent_instance = parent_instance
        self._array_field_name = array_field_name  # For error messages
        self._array_subtype_info = array_type_info.get("subtype")
        if self._array_subtype_info is None:
            raise ValueError(
                f"Array field '{array_field_name}' has no subtype information. type_info={array_type_info}")
        self._array_count = array_type_info.get("count", 0)

        self._element_size = parent_instance._instance_vtype_accessor.get_type_size(
            self._array_subtype_info)
        if self._element_size is None:
            raise ValueError(
                f"Cannot determine element size for array '{array_field_name}'.")

        self._array_start_offset_in_parent = array_start_offset_in_parent

    def _get_element_offset_in_parent_struct(self, index: int) -> int:
        if not 0 <= index < self._array_count:
            raise IndexError(
                f"Array index {index} out of bounds for array '{self._array_field_name}' of size {self._array_count}.")
        return self._array_start_offset_in_parent + (index * self._element_size)

    def __getitem__(self, index: Union[int, slice]) -> Any:
        if isinstance(index, slice):
            start, stop, step = index.indices(self._array_count)
            return [self[i] for i in range(start, stop, step)]
            
        element_offset = self._get_element_offset_in_parent_struct(index)
        return self._parent_instance._read_data(
            self._array_subtype_info,
            element_offset,
            f"{self._array_field_name}[{index}]"
        )

    def __setitem__(self, index: int, value: Any):
        element_offset = self._get_element_offset_in_parent_struct(index)
        self._parent_instance._write_data(
            self._array_subtype_info,
            element_offset,
            value,
            f"{self._array_field_name}[{index}]"
        )
        if self._array_field_name in self._parent_instance._instance_cache:
            del self._parent_instance._instance_cache[self._array_field_name]

    def __len__(self) -> int:
        return self._array_count

    def __iter__(self):
        for i in range(self._array_count):
            yield self[i]

    def __repr__(self) -> str:
        preview_count = min(self._array_count, 3)
        items_preview = [repr(self[i]) for i in range(preview_count)]
        if self._array_count > preview_count:
            items_preview.append("...")
        return f"<BoundArrayView Field='{self._array_field_name}' Count={self._array_count} Items=[{', '.join(items_preview)}]>"

    def __add__(self, offset: int) -> 'Ptr':
        """Decay the array into a pointer and apply arithmetic."""
        if not isinstance(offset, int):
            return NotImplemented
        base_addr = self._parent_instance._instance_offset + self._array_start_offset_in_parent
        return Ptr(
            base_addr + (offset * self._element_size), 
            self._array_subtype_info, 
            self._parent_instance._instance_vtype_accessor
        )

    def __eq__(self, other):
        """Allows `arr == [1, 2, 3]` and comparing arrays to each other."""
        if isinstance(other, list):
            if len(self) != len(other): return False
            return all(self[i] == other[i] for i in range(len(self)))
        if isinstance(other, BoundArrayView):
            if len(self) != len(other): return False
            return all(self[i] == other[i] for i in range(len(self)))
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class BoundTypeInstance:
    """Represents an instance of a DWARF type bound to a memory buffer."""

    def __init__(self, type_name: str, type_def: Union[VtypeUserType, VtypeBaseType, VtypeEnum],
                 buffer: bytearray, vtype_accessor: 'VtypeJson',
                 instance_offset_in_buffer: int = 0):
        if not isinstance(buffer, bytearray):
            raise TypeError("Internal Error: BoundTypeInstance expects a bytearray.")
        self._instance_type_name = type_name
        self._instance_type_def = type_def
        self._instance_buffer = buffer
        self._instance_vtype_accessor = vtype_accessor
        self._instance_offset = instance_offset_in_buffer
        self._instance_cache = {}

    def _get_value(self) -> Any:
        if isinstance(self._instance_type_def, VtypeUserType):
            raise AttributeError(f"'{self._instance_type_name}' is a struct/union and does not have a direct value. Access its fields instead.")
        if isinstance(self._instance_type_def, VtypeBaseType):
            base_type_def = self._instance_type_def
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if base_type_def.size == 0:
                return None
            if compiled_struct_obj is None:
                raise ValueError(f"Cannot get compiled struct for base type '{base_type_def.name}'")
            try:
                return compiled_struct_obj.unpack_from(self._instance_buffer, self._instance_offset)[0]
            except struct.error as e:
                raise struct.error(f"Error unpacking value for base type '{base_type_def.name}': {e}") from e
        elif isinstance(self._instance_type_def, VtypeEnum):
            enum_def = self._instance_type_def
            if enum_def.base is None:
                raise ValueError(f"Enum '{enum_def.name}' has no base type.")
            base_type_def = self._instance_vtype_accessor.get_base_type(enum_def.base)
            if base_type_def is None:
                raise ValueError(f"Base type '{enum_def.base}' for enum '{enum_def.name}' not found.")
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if compiled_struct_obj is None:
                raise ValueError(f"Cannot get compiled struct for enum base type '{enum_def.base}'.")
            try:
                int_val = compiled_struct_obj.unpack_from(self._instance_buffer, self._instance_offset)[0]
                return EnumInstance(enum_def, int_val)
            except struct.error as e:
                raise struct.error(f"Error unpacking value for enum '{enum_def.name}': {e}") from e
        else:
            raise TypeError(f"Cannot get value on internal type: {type(self._instance_type_def).__name__}")

    def _set_value(self, new_value: Any):
        if isinstance(self._instance_type_def, VtypeUserType):
            raise AttributeError(f"Cannot set a direct value on a struct/union '{self._instance_type_name}'. Set its fields instead.")
        if isinstance(self._instance_type_def, VtypeBaseType):
            base_type_def = self._instance_type_def
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if base_type_def.size == 0:
                if new_value is not None:
                    raise ValueError("Cannot assign value to void type.")
                return
            if compiled_struct_obj is None:
                raise ValueError(f"Cannot get compiled struct for base type '{base_type_def.name}' to write value.")
            
            if isinstance(new_value, int):
                new_value = _wrap_integer(new_value, base_type_def.size, bool(base_type_def.signed))
                    
            try:
                compiled_struct_obj.pack_into(self._instance_buffer, self._instance_offset, new_value)
            except struct.error as e:
                raise struct.error(f"Error packing value for base type '{base_type_def.name}': {e}") from e
        elif isinstance(self._instance_type_def, VtypeEnum):
            enum_def = self._instance_type_def
            if enum_def.base is None:
                raise ValueError(f"Enum '{enum_def.name}' has no base type for writing.")
            base_type_def = self._instance_vtype_accessor.get_base_type(enum_def.base)
            if base_type_def is None:
                raise ValueError(f"Base type '{enum_def.base}' for enum '{enum_def.name}' not found for writing.")
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if compiled_struct_obj is None:
                raise ValueError(f"Cannot get compiled struct for enum base type '{enum_def.base}' for writing.")
            
            int_val_to_write: int
            if isinstance(new_value, EnumInstance):
                int_val_to_write = new_value._value
            elif isinstance(new_value, int):
                int_val_to_write = new_value
            elif isinstance(new_value, str):
                found_val = enum_def.constants.get(new_value)
                if found_val is None:
                    raise ValueError(f"Enum constant name '{new_value}' not found in enum '{enum_def.name}'.")
                int_val_to_write = found_val
            else:
                raise TypeError(f"Cannot write type '{type(new_value)}' to enum instance. Expected EnumInstance, int, or str.")
            if isinstance(int_val_to_write, int):
                int_val_to_write = _wrap_integer(int_val_to_write, base_type_def.size, bool(base_type_def.signed))
            try:
                compiled_struct_obj.pack_into(self._instance_buffer, self._instance_offset, int_val_to_write)
            except struct.error as e:
                raise struct.error(f"Error packing value for enum '{enum_def.name}': {e}") from e
        else:
            raise TypeError(f"Setter not applicable to internal type: {type(self._instance_type_def).__name__}")

    def __int__(self) -> int:
        if isinstance(self._instance_type_def, (VtypeBaseType, VtypeEnum)):
            return int(self._get_value())
        if isinstance(self._instance_type_def, VtypeUserType):
            raise TypeError(f"Cannot convert struct/union '{self._instance_type_name}' to int")
        return NotImplemented

    def __index__(self) -> int:
        return self.__int__()

    def __float__(self) -> float:
        if isinstance(self._instance_type_def, VtypeBaseType) and self._instance_type_def.kind == "float":
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
            raise IndexError("Struct/Union index out of range (only [0] is supported for dereferencing)")
        raise TypeError(f"'{self._instance_type_name}' object is not subscriptable")

    def __setitem__(self, index: int, value: Any):
        if index == 0 and isinstance(self._instance_type_def, (VtypeBaseType, VtypeEnum)):
            self._set_value(value)
            return
        if isinstance(self._instance_type_def, VtypeUserType):
            raise TypeError("Cannot overwrite entire struct/union via subscript assignment. Assign to fields instead.")
        raise TypeError(f"'{self._instance_type_name}' object does not support item assignment")

    def _read_data(self, field_type_info: Dict[str, Any], field_offset_in_struct: int, field_name_for_error: str) -> Any:
        kind = field_type_info.get("kind")
        name = field_type_info.get("name")
        absolute_field_offset = self._instance_offset + field_offset_in_struct

        if kind == "base":
            base_type_def = self._instance_vtype_accessor.get_base_type(name)
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if base_type_def.size == 0:
                return None
            return compiled_struct_obj.unpack_from(self._instance_buffer, absolute_field_offset)[0]

        elif kind == "pointer":
            ptr_base_type = self._instance_vtype_accessor.get_base_type("pointer")
            compiled_struct_obj = ptr_base_type.get_compiled_struct()
            address = compiled_struct_obj.unpack_from(self._instance_buffer, absolute_field_offset)[0]
            return Ptr(address, field_type_info.get("subtype"), self._instance_vtype_accessor)

        elif kind == "array":
            return BoundArrayView(self, field_name_for_error, field_type_info, field_offset_in_struct)

        elif kind in ("struct", "union"):
            user_type_def = self._instance_vtype_accessor.get_user_type(name)
            return BoundTypeInstance(name, user_type_def, self._instance_buffer, self._instance_vtype_accessor, absolute_field_offset)

        elif kind == "enum":
            enum_def = self._instance_vtype_accessor.get_enum(name)
            base_type_def = self._instance_vtype_accessor.get_base_type(enum_def.base)
            compiled_struct_obj = base_type_def.get_compiled_struct()
            int_val = compiled_struct_obj.unpack_from(self._instance_buffer, absolute_field_offset)[0]
            return EnumInstance(enum_def, int_val)

        elif kind == "bitfield":
            bit_length = field_type_info.get("bit_length")
            bit_position = field_type_info.get("bit_position")
            underlying_base_name = field_type_info.get("type", {}).get("name")
            underlying_base_def = self._instance_vtype_accessor.get_base_type(underlying_base_name)
            compiled_struct_obj = underlying_base_def.get_compiled_struct()
            storage_unit_val = compiled_struct_obj.unpack_from(self._instance_buffer, absolute_field_offset)[0]
            mask = (1 << bit_length) - 1
            return (storage_unit_val >> bit_position) & mask

        elif kind == "function":
            return f"<FunctionType: {field_type_info.get('name', 'anon_func')}>"
        elif kind == "void" and name == "void":
            return None
        else:
            raise ValueError(f"Unsupported/invalid type kind '{kind}' for field '{field_name_for_error}'.")

    def _write_data(self, field_type_info: Dict[str, Any], field_offset_in_struct: int,
                    value_to_write: Any, field_name_for_error: str):
        kind = field_type_info.get("kind")
        name = field_type_info.get("name")
        absolute_field_offset = self._instance_offset + field_offset_in_struct

        if kind == "base":
            base_type_def = self._instance_vtype_accessor.get_base_type(name)
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if base_type_def.size == 0:
                return
            if base_type_def.signed is False and isinstance(value_to_write, int) and value_to_write < 0:
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
                value_to_write = _wrap_integer(value_to_write, base_type_def.size, bool(base_type_def.signed))
            compiled_struct_obj.pack_into(self._instance_buffer, absolute_field_offset, value_to_write)

        elif kind == "pointer":
            ptr_base_type = self._instance_vtype_accessor.get_base_type("pointer")
            compiled_struct_obj = ptr_base_type.get_compiled_struct()
            address_to_write = value_to_write.address if isinstance(value_to_write, Ptr) else value_to_write
            if isinstance(address_to_write, int):
                address_to_write = _wrap_integer(address_to_write, ptr_base_type.size, False)
            compiled_struct_obj.pack_into(self._instance_buffer, absolute_field_offset, address_to_write)

        elif kind == "enum":
            enum_def = self._instance_vtype_accessor.get_enum(name)
            base_type_def = self._instance_vtype_accessor.get_base_type(enum_def.base)
            compiled_struct_obj = base_type_def.get_compiled_struct()
            if isinstance(value_to_write, EnumInstance):
                int_val_to_write = value_to_write._value
            elif isinstance(value_to_write, int):
                int_val_to_write = value_to_write
            elif isinstance(value_to_write, str):
                int_val_to_write = enum_def.constants.get(value_to_write)
            if isinstance(int_val_to_write, int):
                int_val_to_write = _wrap_integer(int_val_to_write, base_type_def.size, bool(base_type_def.signed))
            compiled_struct_obj.pack_into(self._instance_buffer, absolute_field_offset, int_val_to_write)

        elif kind == "bitfield":
            bit_length = field_type_info.get("bit_length")
            bit_position = field_type_info.get("bit_position")
            underlying_base_name = field_type_info.get("type").get("name")
            underlying_base_def = self._instance_vtype_accessor.get_base_type(underlying_base_name)
            compiled_struct_obj = underlying_base_def.get_compiled_struct()
            current_storage_val = compiled_struct_obj.unpack_from(self._instance_buffer, absolute_field_offset)[0]
            mask = (1 << bit_length) - 1
            value_to_set = value_to_write & mask
            new_storage_val = (current_storage_val & ~(mask << bit_position)) | (value_to_set << bit_position)
            compiled_struct_obj.pack_into(self._instance_buffer, absolute_field_offset, new_storage_val)

        elif kind in ("array", "struct", "union"):
            raise NotImplementedError(f"Direct assignment to field '{field_name_for_error}' of type '{kind}' is not supported.")

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
                
            for f_name, f_def in type_def.fields.items():
                if f_def.anonymous:
                    sub_type = self._instance_vtype_accessor.get_user_type(f_def.type_info.get("name"))
                    if sub_type:
                        res, off = _search(sub_type, target_name, current_offset + f_def.offset)
                        if res:
                            return res, off
            return None, None
            
        return _search(self._instance_type_def, name, 0)

    def __getattr__(self, name: str) -> Any:
        if name.startswith('_instance_') or name.startswith('__'):
            return super().__getattribute__(name)

        if isinstance(self._instance_type_def, VtypeUserType):
            if name in self._instance_cache:
                return self._instance_cache[name]
                
            field_def, field_offset = self._find_field(name)
            if field_def is None:
                raise AttributeError(f"'{self._instance_type_name}' has no attribute '{name}'")
                
            val = self._read_data(field_def.type_info, field_offset, name)
            if field_def.type_info.get("kind") in ["struct", "union", "array"]:
                self._instance_cache[name] = val
            return val

        raise AttributeError(f"Type '{self._instance_type_name}' has no attribute '{name}'. Use '[0]' or cast for base/enum types.")

    def __setattr__(self, name: str, new_value: Any):
        if name.startswith('_instance_') or name.startswith('__'):
            super().__setattr__(name, new_value)
            return

        if isinstance(self._instance_type_def, VtypeUserType):
            field_def, field_offset = self._find_field(name)
            if field_def is None:
                super().__setattr__(name, new_value)
                return
            if field_def.type_info.get("kind") == "array":
                raise NotImplementedError(f"Direct assignment to array field '{name}' is not supported.")
            self._write_data(field_def.type_info, field_offset, new_value, name)
            if name in self._instance_cache:
                del self._instance_cache[name]
            return

        raise AttributeError(f"Cannot set attribute '{name}' on type '{self._instance_type_name}'. Use '[0]' for base/enum types.")
    
    # --- Numeric Magic Methods (For Base Types & Enums) ---
    def _assert_numeric(self):
        if isinstance(self._instance_type_def, VtypeUserType):
            raise TypeError(f"unsupported operand type(s) for struct/union '{self._instance_type_name}'")

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
        if isinstance(self._instance_type_def, VtypeUserType): return NotImplemented
        return self._get_value() < other

    def __le__(self, other):
        if isinstance(self._instance_type_def, VtypeUserType): return NotImplemented
        return self._get_value() <= other

    def __gt__(self, other):
        if isinstance(self._instance_type_def, VtypeUserType): return NotImplemented
        return self._get_value() > other

    def __ge__(self, other):
        if isinstance(self._instance_type_def, VtypeUserType): return NotImplemented
        return self._get_value() >= other

    def __eq__(self, other):
        # 1. Compare against another BoundTypeInstance
        if isinstance(other, BoundTypeInstance):
            # Same memory reference
            if self._instance_buffer is other._instance_buffer and self._instance_offset == other._instance_offset:
                return True
            # Same type name and same exact byte values
            if self._instance_type_name == other._instance_type_name:
                return self.to_bytes() == other.to_bytes()
            # If both are primitive/enum types, try comparing their actual values
            if not isinstance(self._instance_type_def, VtypeUserType) and not isinstance(other._instance_type_def, VtypeUserType):
                return self._get_value() == other._get_value()
            return False
            
        # 2. Compare against native Python types (int, float, str, etc.)
        # If this is a base type or enum, unpack its value and compare natively.
        if not isinstance(self._instance_type_def, VtypeUserType):
            return self._get_value() == other
            
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_bytes(self) -> bytes:
        size = self._instance_type_def.size
        if size == 0:
            return b''
        start = self._instance_offset
        return bytes(self._instance_buffer[start:start + size])

    @property
    def offset(self) -> int:
        return self._instance_offset

    def __repr__(self) -> str:
        return f"<BoundTypeInstance Type='{self._instance_type_name}' Kind='{self._instance_type_def.__class__.__name__}' AtOffset={self._instance_offset}>"

    def __dir__(self):
        attrs = list(super().__dir__())
        if isinstance(self._instance_type_def, VtypeUserType):
            attrs.extend(self._instance_type_def.fields.keys())
        return sorted(list(set(a for a in attrs if a != '_instance_cache')))


class Ptr:
    __slots__ = 'address', '_subtype_info', '_vtype_accessor'

    def __init__(self, address: int, subtype_info: Optional[Dict[str, Any]], vtype_accessor: 'VtypeJson'):
        self.address = address
        self._subtype_info = subtype_info
        self._vtype_accessor = vtype_accessor

    def __repr__(self) -> str:
        subtype_str = "void"
        if self._subtype_info:
            kind, name = self._subtype_info.get("kind"), self._subtype_info.get("name")
            subtype_str = name if name else (kind if kind else "unknown")
        return f"<Ptr ToType='{subtype_str}' Address={self.address:#x}>"

    @property
    def points_to_type_info(self) -> Optional[Dict[str, Any]]: 
        return self._subtype_info

    @property
    def points_to_type_name(self) -> str:
        if not self._subtype_info:
            return "void"
        name, kind = self._subtype_info.get("name"), self._subtype_info.get("kind")
        if name:
            return name
        return "void" if kind == "base" and not name else (kind if kind else "unknown")

    # --- Type Conversions ---
    def __int__(self) -> int: return self.address
    def __index__(self) -> int: return self.address
    def __bool__(self) -> bool: return self.address != 0
    def __hash__(self) -> int: return hash((self.address, self.points_to_type_name))

    # --- Pointer Arithmetic ---
    def __add__(self, offset: int) -> 'Ptr':
        if not isinstance(offset, int):
            return NotImplemented
        size = self._vtype_accessor.get_type_size(self._subtype_info) or 1
        return Ptr(self.address + (offset * size), self._subtype_info, self._vtype_accessor)

    def __sub__(self, other: Union[int, 'Ptr']) -> Union['Ptr', int]:
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
        if isinstance(other, Ptr): return self.address == other.address
        if isinstance(other, int): return self.address == other
        return False

    def __ne__(self, other): return not self.__eq__(other)
    def __lt__(self, other): return self.address < (other.address if isinstance(other, Ptr) else other)
    def __le__(self, other): return self.address <= (other.address if isinstance(other, Ptr) else other)
    def __gt__(self, other): return self.address > (other.address if isinstance(other, Ptr) else other)
    def __ge__(self, other): return self.address >= (other.address if isinstance(other, Ptr) else other)

class EnumInstance:
    __slots__ = '_enum_def', '_value'

    def __init__(self, enum_def: VtypeEnum, value: int):
        self._enum_def = enum_def
        self._value = value

    @property
    def name(self) -> Optional[str]: 
        return self._enum_def.get_name_for_value(self._value)

    def __repr__(self) -> str:
        name_part = f"{self._enum_def.name}.{self.name}" if self.name else f"{self._enum_def.name} (value)"
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