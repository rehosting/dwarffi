import pytest

from dwarffi.dffi import DFFI
from dwarffi.types import VtypeBaseType

# A minimal inline ISF to avoid needing a C compiler or external files for this test
TEST_ISF = {
    "metadata": {"format": "1.4.0"},
    "base_types": {
        "int": {"kind": "int", "size": 4, "signed": True, "endian": "little"},
        "char": {"kind": "char", "size": 1, "signed": True, "endian": "little"},
    },
    "user_types": {
        "Point": {
            "kind": "struct",
            "size": 8,
            "fields": {
                "x": {"type": {"kind": "base", "name": "int"}, "offset": 0},
                "y": {"type": {"kind": "base", "name": "int"}, "offset": 4},
            },
        },
        "Line": {
            "kind": "struct",
            "size": 16,
            "fields": {
                "start": {"type": {"kind": "struct", "name": "Point"}, "offset": 0},
                "end": {"type": {"kind": "struct", "name": "Point"}, "offset": 8},
            },
        },
    },
    "enums": {
        "Status": {
            "size": 4,
            "constants": {"OK": 0, "ERROR": 1, "PENDING": 2},
        }
    },
    "symbols": {},
    "typedefs": {},
}


@pytest.fixture
def ffi():
    """Provides a DFFI instance loaded with the test ISF."""
    return DFFI(isf_input=TEST_ISF)


def test_basetype_call(ffi: DFFI):
    t_int = ffi.get_type("int")
    val = t_int(42)
    assert val == 42


def test_struct_call_dict(ffi: DFFI):
    """Test instantiating a struct using a dictionary argument."""
    t_point = ffi.get_type("Point")
    assert t_point is not None

    p = t_point({"x": 10, "y": 20})
    assert p.x == 10
    assert p.y == 20


def test_struct_call_kwargs(ffi: DFFI):
    """Test instantiating a struct using keyword arguments."""
    t_point = ffi.get_type("Point")
    assert t_point is not None

    p = t_point(x=30, y=40)
    assert p.x == 30
    assert p.y == 40


def test_struct_call_nested_kwargs(ffi: DFFI):
    """Test instantiating a struct with nested dictionaries via kwargs."""
    t_line = ffi.get_type("Line")
    assert t_line is not None

    # This tests the deep_init recursive logic we already had in new()
    line = t_line(start={"x": 1, "y": 2}, end={"x": 3, "y": 4})
    
    assert line.start.x == 1
    assert line.start.y == 2
    assert line.end.x == 3
    assert line.end.y == 4


def test_struct_call_mixed_args_error(ffi: DFFI):
    """Test that mixing positional dictionaries and kwargs raises an error."""
    t_point = ffi.get_type("Point")
    assert t_point is not None

    with pytest.raises(ValueError, match="Cannot mix positional arguments and keyword arguments"):
        t_point({"x": 10}, y=20)


def test_enum_call(ffi: DFFI):
    t_status = ffi.get_type("Status")
    assert t_status is not None

    # status_val is an EnumInstance because of the unboxing in __call__
    status_val = t_status(1)
    
    assert int(status_val) == 1
    assert status_val.name == "ERROR"  # Use the attribute directly!


def test_unbound_type_error():
    """Test that an unbound type correctly refuses to be instantiated."""
    # Create a raw type not fetched from DFFI
    raw_int = VtypeBaseType(name="int", size=4, kind="int")
    
    with pytest.raises(RuntimeError, match="not bound to a DFFI engine"):
        raw_int(10)