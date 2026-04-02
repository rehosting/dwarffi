from dwarffi.dffi import DFFI

def test_unions_and_casting():
    d = DFFI()
    d.cdef("""
        typedef unsigned char u8;  // Add this to make d.t.u8 work!
        
        union IPAddress {
            unsigned int addr;
            u8 octets[4];
        };
        
        struct Packet {
            unsigned short protocol;
            union IPAddress src;
            union IPAddress dst;
        };
    """)

    pkt = d.t.Packet(
        protocol=2048,
        src={'octets': [192, 168, 1, 1]},
        dst={'addr': 0x01020304}
    )

    # ... (assertions) ...

    # 3. Reinterpretation Casting
    # Now d.t.u8 will be found correctly
    view = d.cast(d.t.u8.array(4), pkt.src)
    assert list(view) == [192, 168, 1, 1]

    # 4. In-place modification through the casted view
    view[0] = 10
    assert pkt.src.octets[0] == 10

    print("Example 05 (Unions & Casting): Success")

if __name__ == "__main__":
    test_unions_and_casting()