from dwarffi.dffi import DFFI

def test_bitfields():
    d = DFFI()
    # Define a classic TCP-style header or hardware register
    d.cdef("""
        struct ControlRegister {
            unsigned int enabled  : 1;
            unsigned int mode     : 3;
            unsigned int priority : 4;
            unsigned int reserved : 24;
        };
    """)

    # 1. Instantiate via call sugar
    reg = d.t.ControlRegister(enabled=1, mode=5, priority=10)

    # 2. Verify bit-level isolation
    assert reg.enabled == 1
    assert reg.mode == 5
    assert reg.priority == 10
    
    # 3. Verify that writing one field doesn't corrupt neighbors
    reg.mode = 0
    assert reg.enabled == 1   # Still enabled
    assert reg.priority == 10 # Still priority 10
    
    # 4. Verify physical layout (4 bytes total for 32 bits)
    assert d.sizeof(reg) == 4
    
    # Verify raw bytes (little-endian: 1 | (0 << 1) | (10 << 4) == 0xA1)
    raw = d.to_bytes(reg)
    assert raw[0] == 0xA1 

    print("Example 04 (Bitfields): Success")

if __name__ == "__main__":
    test_bitfields()