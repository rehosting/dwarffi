import json
import lzma
from pathlib import Path
from collections import deque

TYPE_KINDS_WITH_SUBTYPE = {"pointer", "array"}
TYPE_KINDS_WITH_NAME = {"struct", "class", "union", "enum", "base"}

def load_isf(path: Path) -> dict:
    if path.suffixes[-2:] == [".json", ".xz"] or path.suffix == ".xz":
        with lzma.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text("utf-8"))

def collect_type_refs(desc: dict) -> list[dict]:
    """Return nested type descriptors referenced by this descriptor."""
    out = []
    if not isinstance(desc, dict):
        return out
    kind = desc.get("kind")
    if kind in TYPE_KINDS_WITH_SUBTYPE and "subtype" in desc:
        out.append(desc["subtype"])
    if kind == "bitfield" and "type" in desc:
        out.append(desc["type"])
    return out

def type_key(desc: dict) -> tuple[str, str] | None:
    kind = desc.get("kind")
    if kind in {"struct", "class", "union"}:
        return ("user_types", desc["name"])
    if kind == "enum":
        return ("enums", desc["name"])
    if kind == "base":
        return ("base_types", desc["name"])
    return None
def minify_isf(isf: dict, root_structs: list[str], max_symbols: int = 0) -> dict:
    base_types = isf.get("base_types", {})
    user_types = isf.get("user_types", {})
    enums = isf.get("enums", {})
    symbols = isf.get("symbols", {})

    keep_base = set()
    keep_user = set()
    keep_enum = set()

    q = deque()
    for s in root_structs:
        q.append({"kind": "struct", "name": s})

    while q:
        td = q.popleft()
        if not isinstance(td, dict):
            continue

        kind = td.get("kind")

        # ✅ IMPORTANT: our runtime expects a base type named "pointer"
        # whenever pointer kinds appear in the graph.
        if kind == "pointer":
            keep_base.add("pointer")
            if "subtype" in td:
                q.append(td["subtype"])
            continue

        # (Optional but usually good hygiene)
        if kind == "void":
            keep_base.add("void")
            continue

        k = type_key(td)
        if not k:
            for child in collect_type_refs(td):
                q.append(child)
            continue

        section, name = k
        if section == "base_types":
            keep_base.add(name)

        elif section == "enums":
            if name in keep_enum:
                continue
            keep_enum.add(name)
            enum_obj = enums.get(name)
            if enum_obj and "base" in enum_obj:
                keep_base.add(enum_obj["base"])

        elif section == "user_types":
            if name in keep_user:
                continue
            keep_user.add(name)
            ut = user_types.get(name)
            if not ut:
                continue
            for field in ut.get("fields", {}).values():
                q.append(field["type"])

        for child in collect_type_refs(td):
            q.append(child)

    # ✅ Also: ensure required base types exist if referenced
    required = {"pointer"}
    for r in required:
        if r in base_types:
            keep_base.add(r)

    kept_symbols = {}
    if max_symbols > 0:
        for i, (sym, obj) in enumerate(symbols.items()):
            if i >= max_symbols:
                break
            kept_symbols[sym] = obj

    return {
        "metadata": isf.get("metadata", {}),
        "base_types": {k: base_types[k] for k in keep_base if k in base_types},
        "user_types": {k: user_types[k] for k in keep_user if k in user_types},
        "enums": {k: enums[k] for k in keep_enum if k in enums},
        "symbols": kept_symbols,
    }

def main():
    src = Path("/home/luke/workspace/dwarffi/windows/ntkrnlmp.pdb/37144E973CF046B9920BE37E36A8B458-2.json.xz")
    dst = Path("tests/fixtures/windows_min_ntkrnlmp.json")

    isf = load_isf(src)
    slim = minify_isf(isf, root_structs=["_LIST_ENTRY", "_UNICODE_STRING"], max_symbols=0)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(slim, indent=2), "utf-8")
    print("Wrote:", dst, "bytes:", dst.stat().st_size)

if __name__ == "__main__":
    main()