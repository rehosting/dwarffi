from dwarffi.dffi import DFFI

def main():
    # 1. Initialize the engine
    d = DFFI()

    # 2. Compile C code on the fly to extract type information!
    # (Requires gcc/clang to be installed on the host system)
    d.cdef("""
        typedef unsigned int u32;
        
        struct Point {
            int x;
            int y;
        };

        struct Player {
            u32 health;
            struct Point position;
            struct Point* target;
        };
    """)

    # 3. Instantiate types using the new `d.t` namespace and `__call__` sugar
    # We can use nested keyword arguments to cleanly initialize structs
    p1 = d.t.Point(x=10, y=20)
    p2 = d.t.Point(x=100, y=200)
    
    player = d.t.Player(
        health=100,
        position=p1,
    )

    print(f"Player Health: {player.health}")
    print(f"Player X Pos: {player.position.x}")

    # 4. Pointers and Casting
    # We can get a pointer type dynamically using `.ptr`
    point_ptr_t = d.t.Point.ptr
    
    # Cast an absolute memory address (e.g., 0x4000) to a Point pointer
    ptr = d.cast(point_ptr_t, 0x4000)
    print(f"Pointer Address: {hex(ptr.address)}")
    
    # Pointer arithmetic (automatically scales by sizeof(Point), which is 8 bytes)
    next_ptr = ptr + 2
    print(f"Advanced Pointer: {hex(next_ptr.address)}") # 0x4010

    # 5. Arrays and Slicing
    # Create an array type of 5 integers (returns an ISF dict)
    int_arr_t = d.t.int.array(5)
    
    # Instantiate it using d.new() 
    arr = d.new(int_arr_t, [10, 20, 30, 40, 50])
    
    # Pythonic slicing works natively!
    print(f"Array Slice: {arr[1:4]}") # [20, 30, 40]

if __name__ == "__main__":
    main()