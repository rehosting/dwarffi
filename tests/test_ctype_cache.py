from dwarffi import DFFI


def _minimal_isf(endian: str = "little", ptr_size: int = 8) -> dict:
    return {
        "metadata": {},
        "base_types": {
            "int": {"kind": "int", "size": 4, "signed": True, "endian": endian},
            "pointer": {"kind": "pointer", "size": ptr_size, "endian": endian},
            "void": {"kind": "void", "size": 0, "signed": False, "endian": endian},
        },
        "user_types": {},
        "enums": {},
        "symbols": {},
    }


def test_parse_ctype_lru_cache_is_bounded_and_hits_increment():
    ffi = DFFI(_minimal_isf())

    info0 = ffi._parse_ctype_string.cache_info()
    assert info0.maxsize == 2048
    assert info0.hits == 0
    assert info0.misses == 0

    # One typeof() may trigger multiple internal parses; require "some" cache population.
    ffi.typeof("int *")
    info1 = ffi._parse_ctype_string.cache_info()

    miss_delta = info1.misses - info0.misses
    assert miss_delta >= 1
    assert info1.currsize >= 1

    # Repeating should now yield cache hits (again, potentially multiple).
    ffi.typeof("int *")
    info2 = ffi._parse_ctype_string.cache_info()

    hit_delta = info2.hits - info1.hits
    assert hit_delta >= 1

    # Should not introduce new misses when repeating the exact same query,
    # unless the implementation intentionally rekeys/normalizes differently.
    # This is a strong-but-still-realistic invariant: in your observed behavior,
    # the repeat does not add misses.
    assert info2.misses == info1.misses


def test_parse_ctype_cache_is_per_instance_not_global():
    ffi1 = DFFI(_minimal_isf())
    ffi2 = DFFI(_minimal_isf())

    # Warm ffi1 cache
    ffi1.typeof("int *")
    ffi1.typeof("int *")
    info1 = ffi1._parse_ctype_string.cache_info()
    assert info1.hits >= 1

    # ffi2 should start cold (independent cache object)
    info2_before = ffi2._parse_ctype_string.cache_info()
    assert info2_before.hits == 0
    assert info2_before.misses == 0

    ffi2.typeof("int *")
    info2_after = ffi2._parse_ctype_string.cache_info()

    # Cold instance should incur misses (often >1) and no hits on first call
    assert info2_after.misses >= 1
    assert info2_after.hits == 0