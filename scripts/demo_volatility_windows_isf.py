#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from rich import print
from dwarffi import DFFI


DEFAULT_TYPES = [
    "_EPROCESS",
    "_ETHREAD",
    "_KPCR",
    "_KPRCB",
    "_LIST_ENTRY",
    "_UNICODE_STRING",
]


def _fmt_hex(x: int, width_bytes: int) -> str:
    mask = (1 << (width_bytes * 8)) - 1
    return hex(x & mask)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="inspect_windows_isf",
        description=(
            "Inspect a Volatility-style Windows ISF (.json or .json.xz) using dwarffi.\n"
            "This demonstrates type resolution, layout inspection, and instance formatting/serialization."
        ),
    )
    parser.add_argument(
        "isf_path",
        type=Path,
        help="Path to a Windows ISF file (either .json or .json.xz).",
    )
    parser.add_argument(
        "--types",
        nargs="*",
        default=DEFAULT_TYPES,
        help="List of Windows types (struct names) to probe, e.g. _EPROCESS _UNICODE_STRING.",
    )
    parser.add_argument(
        "--layout",
        default="_UNICODE_STRING",
        help="Struct type name to pahole-style inspect (default: _UNICODE_STRING).",
    )
    parser.add_argument(
        "--demo",
        choices=["list_entry", "unicode_string", "none"],
        default="unicode_string",
        help="Which instance demo to run (default: unicode_string).",
    )
    args = parser.parse_args(argv)

    if not args.isf_path.exists():
        print(f"[error] File not found: {args.isf_path}", file=sys.stderr)
        return 2

    print(f"[1/4] Loading ISF: {args.isf_path}")

    # Per your note: DFFI can accept json/json.xz paths directly.
    ffi = DFFI(str(args.isf_path))

    # Try to access underlying metadata if DFFI exposes it; otherwise, fall back gracefully.
    isf = getattr(ffi, "isf", None) or getattr(ffi, "_isf", None)

    print(f"[2/4] Inspecting metadata (Windows PDB identity, if present)")
    if isinstance(isf, dict):
        md = isf.get("metadata", {})
        pdb = md.get("windows", {}).get("pdb", {})
        if pdb:
            print(f"      PDB database    : {pdb.get('database')}")
            print(f"      PDB GUID / age  : {pdb.get('GUID')} / {pdb.get('age')}")
            print(f"      PDB machine_type: {pdb.get('machine_type')}")
        else:
            print("      No metadata.windows.pdb block found in this ISF.")
    else:
        print("      (Metadata not directly exposed by DFFI instance; skipping.)")

    # Determine pointer size (use DFFI’s understanding, not assumptions)
    ptr_size = None
    try:
        # Most reliable: the base type "pointer"
        ptr_bt = ffi.get_base_type("pointer")
        ptr_size = ptr_bt.size if ptr_bt else None
    except Exception:
        ptr_size = None

    if ptr_size:
        print(f"      Pointer width   : {ptr_size * 8}-bit")
    else:
        print("      Pointer width   : (unknown; will still attempt demos)")

    print(f"[3/4] Probing common Windows structs to demonstrate type resolution")
    for tname in args.types:
        try:
            t = ffi.typeof(f"struct {tname}")
            print(f"      struct {tname:<16} sizeof = {ffi.sizeof(t)}")
        except Exception as e:
            print(f"      struct {tname:<16} not found ({e})")

    # Layout inspection: pahole-style
    print(f"\n[4/4] Layout inspection (pahole-style): struct {args.layout}")
    try:
        ffi.inspect_layout(f"struct {args.layout}")
    except Exception as e:
        print(f"      Layout inspection failed: {e}")

    # Instance demo: show pretty_print + to_dict for something realistic
    if args.demo == "none":
        return 0

    if args.demo == "unicode_string":
        print("\n[demo] Instance demo: struct _UNICODE_STRING")
        try:
            us = ffi.typeof("struct _UNICODE_STRING")
            buf = bytearray(ffi.sizeof(us))
            inst = ffi.from_buffer("struct _UNICODE_STRING", buf)

            # Fill common fields if present
            if hasattr(inst, "Length"):
                inst.Length = 8
            if hasattr(inst, "MaximumLength"):
                inst.MaximumLength = 16
            if hasattr(inst, "Buffer"):
                raw = 0x1122334455667788
                inst.Buffer = raw
                if ptr_size:
                    print(f"      Buffer assigned : {hex(raw)} (will be truncated to {_fmt_hex(raw, ptr_size)} if 32-bit)")
                else:
                    print(f"      Buffer assigned : {hex(raw)}")

            print("\n      pretty_print():")
            print(ffi.pretty_print(inst))

            print("\n      to_dict():")
            print(json.dumps(ffi.to_dict(inst), indent=2))

        except Exception as e:
            print(f"      _UNICODE_STRING demo failed: {e}")

    elif args.demo == "list_entry":
        print("\n[demo] Instance demo: struct _LIST_ENTRY")
        try:
            le = ffi.typeof("struct _LIST_ENTRY")
            buf = bytearray(ffi.sizeof(le))
            inst = ffi.from_buffer("struct _LIST_ENTRY", buf)

            raw_f = 0xAABBCCDDEEFF0011
            raw_b = 0x1100FFEEDDCCBBAA

            if hasattr(inst, "Flink"):
                inst.Flink = raw_f
            if hasattr(inst, "Blink"):
                inst.Blink = raw_b

            if ptr_size:
                print(f"      Flink assigned  : {hex(raw_f)} -> stored {_fmt_hex(raw_f, ptr_size)}")
                print(f"      Blink assigned  : {hex(raw_b)} -> stored {_fmt_hex(raw_b, ptr_size)}")

            print("\n      pretty_print():")
            print(ffi.pretty_print(inst))

            print("\n      to_dict():")
            print(json.dumps(ffi.to_dict(inst), indent=2))

        except Exception as e:
            print(f"      _LIST_ENTRY demo failed: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())