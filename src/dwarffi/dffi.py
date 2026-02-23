import weakref
from typing import Any, Dict, Optional, Union, List
from dwarffi.core import (
    VtypeJsonGroup, 
    load_isf_json, 
    BoundTypeInstance, 
    VtypeUserType, 
    VtypeBaseType, 
    VtypeEnum,
    Ptr,
    BoundArrayView
)

class DFFI:
    def __init__(self, isf_path: Optional[str] = None):
        self._isf_group = VtypeJsonGroup([])
        if isf_path:
            self.load_isf(isf_path)
            
    def load_isf(self, path: str):
        if path not in self._isf_group.vtypejsons:
            self._isf_group._file_order.append(path)
            self._isf_group.vtypejsons[path] = load_isf_json(path)

    def typeof(self, ctype: Union[str, BoundTypeInstance, Ptr]) -> Union[VtypeUserType, VtypeBaseType, VtypeEnum, Dict]:
        if isinstance(ctype, BoundTypeInstance):
            return ctype._instance_type_def
        if isinstance(ctype, Ptr):
            return ctype.points_to_type_info
            
        if isinstance(ctype, str):
            t = self._isf_group.get_type(ctype)
            if not t:
                # Handle basic pointer types dynamically if needed
                if ctype.endswith("*"):
                    base_name = ctype[:-1].strip()
                    return {"kind": "pointer", "subtype": {"name": base_name}}
                raise KeyError(f"Unknown DWARF type: '{ctype}'")
            return t
            
        raise TypeError(f"Expected string or BoundTypeInstance, got {type(ctype)}")

    def sizeof(self, ctype: Union[str, BoundTypeInstance, Any]) -> int:
        if isinstance(ctype, BoundTypeInstance):
            size = ctype._instance_type_def.size
        elif isinstance(ctype, str):
            if ctype.endswith("*"):
                return self._isf_group.get_base_type("pointer").size
            size = self.typeof(ctype).size
        elif hasattr(ctype, "size"):
            size = ctype.size
        else:
            raise TypeError(f"Cannot determine size of {ctype}")
            
        if size is None:
            raise ValueError(f"Type '{ctype}' has an unknown or undefined size.")
        return size

    def offsetof(self, ctype: str, *fields_or_indexes) -> int:
        t = self.typeof(ctype)
        if not isinstance(t, VtypeUserType):
            raise TypeError(f"Type '{ctype}' is not a struct or union.")
            
        offset = 0
        current_type = t
        
        for field_name in fields_or_indexes:
            if isinstance(current_type, VtypeUserType):
                field = current_type.fields.get(field_name)
                if not field:
                    raise KeyError(f"Type '{current_type.name}' has no field '{field_name}'")
                if field.offset is None:
                    raise ValueError(f"Field '{field_name}' has an unknown offset.")
                offset += field.offset
                
                # Update current_type for nested structures
                if field.type_info.get("kind") in ["struct", "union"]:
                    current_type = self._isf_group.get_type(field.type_info.get("name"))
            else:
                raise TypeError(f"Cannot get offset of '{field_name}' inside non-struct type.")
                
        return offset

    def addressof(self, cdata: BoundTypeInstance, *fields_or_indexes) -> int:
        """
        Returns the offset/address of the buffer. In a rehosting context, 
        this represents the local buffer offset unless mapped via a memory backend.
        """
        base_addr = cdata._instance_offset
        if fields_or_indexes:
            # We reuse offsetof logic by passing the type name
            return base_addr + self.offsetof(cdata._instance_type_name, *fields_or_indexes)
        return base_addr

    def new(self, ctype: str, init: Any = None) -> BoundTypeInstance:
        t = self.typeof(ctype)
        if getattr(t, 'size', None) is None:
            raise ValueError(f"Cannot allocate memory for type '{ctype}' with unknown size.")
            
        buf = bytearray(t.size)
        instance = self._isf_group.create_instance(t, buf)
        
        if init is not None:
            if isinstance(init, dict):
                for k, v in init.items():
                    setattr(instance, k, v)
            elif isinstance(init, (int, str)):
                instance._value = init
            else:
                raise TypeError(f"Unsupported initializer type: {type(init)}")
                
        return instance

    def cast(self, ctype: str, value: Any) -> Union[BoundTypeInstance, Ptr]:
        """
        Interpret 'value' as the specified C type.
        """
        t = self.typeof(ctype)
        
        # Casting an integer to a pointer
        if isinstance(value, int):
            if isinstance(t, dict) and t.get("kind") == "pointer":
                return Ptr(value, t.get("subtype"), self._isf_group)
            
            # Casting an int to a primitive type (allocates a new detached wrapper)
            buf = bytearray(getattr(t, 'size', 8))
            instance = self._isf_group.create_instance(t, buf)
            instance._value = value
            return instance

        # Re-casting an existing buffer to a new type
        if isinstance(value, BoundTypeInstance):
            return self._isf_group.create_instance(t, value._instance_buffer, value._instance_offset)

        raise TypeError(f"Cannot cast {type(value)} to {ctype}")

    def from_buffer(self, ctype: str, python_buffer: Union[bytearray, memoryview, bytes], require_writable: bool = False) -> BoundTypeInstance:
        if require_writable and isinstance(python_buffer, bytes):
            raise TypeError("Buffer is read-only")
            
        t = self.typeof(ctype)
        # Convert bytes to bytearray to satisfy BoundTypeInstance requirement if read-only is tolerated
        if isinstance(python_buffer, bytes):
            python_buffer = bytearray(python_buffer)
            
        return self._isf_group.create_instance(t, python_buffer)

    def buffer(self, cdata: BoundTypeInstance, size: Optional[int] = None) -> memoryview:
        """Returns a memoryview over the underlying bytearray of the cdata."""
        size = size if size is not None else cdata._instance_type_def.size
        start = cdata._instance_offset
        return memoryview(cdata._instance_buffer)[start:start+size]

    def memmove(self, dest: Union[BoundTypeInstance, bytearray], src: Union[BoundTypeInstance, bytearray, bytes], n: int):
        """Copy n bytes from src to dest."""
        dest_buf = dest._instance_buffer if isinstance(dest, BoundTypeInstance) else dest
        dest_off = dest._instance_offset if isinstance(dest, BoundTypeInstance) else 0
        
        src_buf = src._instance_buffer if isinstance(src, BoundTypeInstance) else src
        src_off = src._instance_offset if isinstance(src, BoundTypeInstance) else 0

        dest_buf[dest_off:dest_off+n] = src_buf[src_off:src_off+n]

    def string(self, cdata: Union[BoundTypeInstance, Ptr], maxlen: int = 4096) -> bytes:
        if isinstance(cdata, BoundTypeInstance):
            raw_bytes = cdata.to_bytes()
            if maxlen > 0:
                raw_bytes = raw_bytes[:maxlen]
            null_idx = raw_bytes.find(b'\x00')
            return raw_bytes[:null_idx] if null_idx != -1 else raw_bytes
            
        if isinstance(cdata, Ptr):
            # Without a memory backend plugged in, a raw Ptr can't be dereferenced for string reading.
            raise NotImplementedError("Cannot read string from raw Ptr address without an attached memory backend.")

    def unpack(self, cdata: Union[BoundArrayView, BoundTypeInstance], length: int) -> list:
        if isinstance(cdata, BoundArrayView):
            return [cdata[i] for i in range(min(length, len(cdata)))]
        raise TypeError("unpack() currently requires an array view.")

    def gc(self, cdata: BoundTypeInstance, destructor: callable, size: int = 0) -> BoundTypeInstance:
        """
        Attach a finalizer to the cdata object. When garbage collected, 
        destructor(cdata) will be executed.
        """
        weakref.finalize(cdata, destructor, cdata)
        return cdata