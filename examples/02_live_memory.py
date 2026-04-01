from dwarffi.dffi import DFFI
from dwarffi.backend import MemoryBackend

# A simple mock backend that simulates reading/writing to a target's memory
class MockDebuggerBackend(MemoryBackend):
    def __init__(self):
        self.memory = bytearray(0x10000) # 64KB mock RAM
        
    def read(self, address: int, size: int) -> bytes:
        return bytes(self.memory[address : address + size])
        
    def write(self, address: int, data: bytes) -> None:
        self.memory[address : address + len(data)] = data

def main():
    # 1. Initialize our memory proxy
    target_ram = MockDebuggerBackend()
    
    # 2. Load DFFI with the backend attached
    d = DFFI(backend=target_ram)
    
    # Define some types
    d.cdef("""
        struct Node {
            int value;
            struct Node* next;
        };
    """)
    
    # 3. Create objects directly inside the target's RAM!
    # We write a Node at 0x1000, and another at 0x1020
    node1 = d.from_address("struct Node", 0x1000)
    node2 = d.from_address("struct Node", 0x1020)
    
    # Standard Python assignments translate to live memory writes over the backend
    node1.value = 42
    node1.next = 0x1020  # Link to the second node
    
    node2.value = 99
    node2.next = 0x0
    
    # 4. Traverse the linked list natively
    print("Traversing Linked List in Target Memory:")
    
    current_node_ptr = d.cast(d.t.Node.ptr, 0x1000)
    
    while current_node_ptr:
        # .deref() reads from the backend to construct the Python object
        node = current_node_ptr.deref()
        print(f"Node at {hex(current_node_ptr.address)}: value = {node.value}")
        
        # Advance the pointer
        current_node_ptr = node.next

if __name__ == "__main__":
    main()