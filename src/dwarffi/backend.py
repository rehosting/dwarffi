import sys
from typing import Union


class MemoryBackend:
    def read(self, address: int, size: int) -> bytes:
        raise NotImplementedError("Backend does not support reading memory.")

    def write(self, address: int, data: bytes) -> None:
        raise NotImplementedError("Backend does not support writing memory.")

class BytesBackend(MemoryBackend):
    def __init__(self, data: Union[bytes, bytearray]):
        self._data = bytearray(data)
        self._size = len(self._data)

    def read(self, address: int, size: int) -> bytes:
        if address < 0 or address + size > self._size:
            raise MemoryError(f"Read at 0x{address:x} for {size} bytes is out of bounds.")
        return bytes(self._data[address : address + size])

    def write(self, address: int, data: bytes) -> None:
        end = address + len(data)
        if address < 0 or end > self._size:
            raise MemoryError(f"Write at 0x{address:x} for {len(data)} bytes is out of bounds.")
        self._data[address : end] = data

class LiveMemoryProxy:
    """
    Acts like a Python bytearray but delegates all slice accesses to the MemoryBackend.
    This allows BoundTypeInstance to remain completely ignorant of backends and operate 
    directly on live data via standard indexing.
    """
    def __init__(self, backend: MemoryBackend):
        self.backend = backend

    def __getitem__(self, key: Union[int, slice]) -> bytes:
        if isinstance(key, slice):
            start = key.start or 0
            stop = key.stop
            if stop is None:
                raise ValueError("LiveMemoryProxy requires bounded slices.")
            return self.backend.read(start, stop - start)
        elif isinstance(key, int):
            return self.backend.read(key, 1)
        raise TypeError("Invalid index type")

    def __setitem__(self, key: Union[int, slice], value: bytes) -> None:
        if isinstance(key, slice):
            start = key.start or 0
            self.backend.write(start, value)
        elif isinstance(key, int):
            self.backend.write(key, value)
        else:
            raise TypeError("Invalid index type")
            
    def __len__(self) -> int:
        return sys.maxsize  # Arbitrarily large to bypass local bounds checks