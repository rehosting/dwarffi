from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pytest

from dwarffi import DFFI, VtypeJson


GHIDRA_SCRIPT_DIR = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dwarffi"
    / "ghidra_scripts"
)
EXPORT_SCRIPT_PATH = GHIDRA_SCRIPT_DIR / "Ghidra2ISF.java"
IMPORT_SCRIPT_PATH = GHIDRA_SCRIPT_DIR / "ISF2Ghidra.java"


def _sample_full_profile_isf() -> dict:
    return {
        "metadata": {
            "producer": {"name": "ghidra2isf", "version": "0.1.0"},
            "format": "6.2.0",
        },
        "base_types": {
            "void": {"size": 0, "signed": False, "kind": "void", "endian": "little"},
            "pointer": {"size": 8, "signed": False, "kind": "pointer", "endian": "little"},
            "uint8_t": {"size": 1, "signed": False, "kind": "int", "endian": "little"},
            "uint16_t": {"size": 2, "signed": False, "kind": "int", "endian": "little"},
            "uint32_t": {"size": 4, "signed": False, "kind": "int", "endian": "little"},
            "int": {"size": 4, "signed": True, "kind": "int", "endian": "little"},
        },
        "user_types": {
            "Inner": {
                "size": 4,
                "kind": "struct",
                "fields": {
                    "a": {"offset": 0, "type": {"kind": "base", "name": "uint16_t"}},
                    "b": {"offset": 2, "type": {"kind": "base", "name": "uint8_t"}},
                },
            },
            "Value": {
                "size": 4,
                "kind": "union",
                "fields": {
                    "word": {"offset": 0, "type": {"kind": "base", "name": "uint32_t"}},
                    "bytes": {
                        "offset": 0,
                        "type": {
                            "kind": "array",
                            "count": 4,
                            "subtype": {"kind": "base", "name": "uint8_t"},
                        },
                    },
                },
            },
            "Packet": {
                "size": 16,
                "kind": "struct",
                "fields": {
                    "id": {"offset": 0, "type": {"kind": "typedef", "name": "my_u32"}},
                    "inner": {"offset": 4, "type": {"kind": "struct", "name": "Inner"}},
                    "value": {"offset": 8, "type": {"kind": "union", "name": "Value"}},
                    "flags": {
                        "offset": 12,
                        "type": {
                            "kind": "bitfield",
                            "bit_length": 3,
                            "bit_position": 0,
                            "type": {"kind": "base", "name": "uint8_t"},
                        },
                    },
                },
            },
        },
        "enums": {
            "Color": {
                "size": 4,
                "base": "int",
                "constants": {"RED": 1, "BLUE": 2},
            }
        },
        "symbols": {
            "global_counter": {
                "address": 0x4010,
                "type": {"kind": "base", "name": "int"},
            }
        },
        "functions": {
            "add_packet": {
                "address": 0x1000,
                "return_type": {"kind": "base", "name": "int"},
                "parameters": [
                    {
                        "name": "p",
                        "type": {
                            "kind": "pointer",
                            "subtype": {"kind": "struct", "name": "Packet"},
                        },
                    },
                    {"name": "x", "type": {"kind": "base", "name": "int"}},
                ],
            }
        },
        "typedefs": {
            "my_u32": {"kind": "base", "name": "uint32_t"},
        },
    }


def test_ghidra2isf_script_is_bundled() -> None:
    source = EXPORT_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "public class Ghidra2ISF extends GhidraScript" in source
    assert '"base_types"' in source
    assert '"functions"' in source
    assert "--types-only" in source


def test_isf2ghidra_script_is_bundled() -> None:
    source = IMPORT_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "public class ISF2Ghidra extends GhidraScript" in source
    assert 'new CategoryPath("/ISF")' in source
    assert "importTypedefs" in source
    assert "--no-functions" in source


def test_ghidra2isf_full_profile_shape_loads_in_dwarffi(tmp_path: Path) -> None:
    isf_path = tmp_path / "ghidra_sample.isf.json"
    isf_path.write_text(json.dumps(_sample_full_profile_isf()), encoding="utf-8")

    parsed = VtypeJson(str(isf_path))
    assert parsed.get_type("Packet") is not None
    assert parsed.get_enum("Color") is not None
    assert parsed.get_symbol("global_counter") is not None
    assert parsed.get_function("add_packet") is not None

    ffi = DFFI(str(isf_path))
    assert ffi.sizeof("Packet") == 16
    assert ffi.sizeof("my_u32") == 4


def _download_ghidra(cache_dir: Path) -> Path:
    version = os.environ.get("DFFI_GHIDRA_VERSION", "12.1_PUBLIC_20260513")
    build = os.environ.get("DFFI_GHIDRA_BUILD", "Ghidra_12.1_build")
    dirname = f"ghidra_{version}"
    install_dir = cache_dir / dirname
    if (install_dir / "support" / "analyzeHeadless").exists():
        _ensure_executable(install_dir / "support" / "analyzeHeadless")
        return install_dir
    for candidate in cache_dir.glob("ghidra_*"):
        if (candidate / "support" / "analyzeHeadless").exists():
            _ensure_executable(candidate / "support" / "analyzeHeadless")
            return candidate

    url = os.environ.get(
        "DFFI_GHIDRA_URL",
        f"https://github.com/NationalSecurityAgency/ghidra/releases/download/{build}/{dirname}.zip",
    )
    zip_path = cache_dir / f"{dirname}.zip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(cache_dir)
    if (install_dir / "support" / "analyzeHeadless").exists():
        _ensure_executable(install_dir / "support" / "analyzeHeadless")
        return install_dir
    for candidate in cache_dir.glob("ghidra_*"):
        if (candidate / "support" / "analyzeHeadless").exists():
            _ensure_executable(candidate / "support" / "analyzeHeadless")
            return candidate
    raise FileNotFoundError("Downloaded Ghidra archive did not contain support/analyzeHeadless")


def _ensure_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    for helper in path.parent.glob("*.sh"):
        helper.chmod(helper.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _ghidra_home() -> Path | None:
    env_home = os.environ.get("GHIDRA_HOME")
    if env_home:
        return Path(env_home)

    analyze_headless = shutil.which("analyzeHeadless")
    if analyze_headless:
        return Path(analyze_headless).resolve().parents[1]

    if os.environ.get("DFFI_GHIDRA_DOWNLOAD") == "1":
        return _download_ghidra(Path(os.environ.get("DFFI_GHIDRA_CACHE", "/tmp/dwarffi-ghidra")))

    return None


@pytest.mark.skipif(
    os.environ.get("DFFI_GHIDRA_TEST") != "1",
    reason="set DFFI_GHIDRA_TEST=1 to run the Ghidra integration test",
)
def test_ghidra2isf_exports_real_program_with_analyze_headless(tmp_path: Path) -> None:
    ghidra_home = _ghidra_home()
    if ghidra_home is None:
        pytest.skip("GHIDRA_HOME/analyzeHeadless not found; set DFFI_GHIDRA_DOWNLOAD=1 to download")

    analyze_headless = ghidra_home / "support" / "analyzeHeadless"
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc is required for the Ghidra integration fixture")

    source_path = tmp_path / "fixture.c"
    binary_path = tmp_path / "fixture"
    isf_path = tmp_path / "fixture.isf.json"
    project_dir = tmp_path / "ghidra_project"
    ghidra_user_home = tmp_path / "ghidra_home"
    source_path.write_text(
        """
        #include <stdint.h>

        enum Color { RED = 1, BLUE = 2 };
        typedef uint32_t my_u32;

        struct Inner {
            uint16_t a;
            uint8_t b;
        };

        union Value {
            uint32_t word;
            uint8_t bytes[4];
        };

        struct Packet {
            my_u32 id;
            struct Inner inner;
            union Value value;
            uint8_t flags;
        };

        int global_counter = 7;

        int add_packet(struct Packet *p, int x) {
            return (int)p->id + x + global_counter;
        }
        """,
        encoding="utf-8",
    )

    subprocess.run(
        [
            gcc,
            "-g",
            "-O0",
            "-fno-eliminate-unused-debug-types",
            "-c",
            str(source_path),
            "-o",
            str(binary_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    env = os.environ.copy()
    env["HOME"] = str(ghidra_user_home)
    env["XDG_CONFIG_HOME"] = str(ghidra_user_home / ".config")
    env["XDG_CACHE_HOME"] = str(ghidra_user_home / ".cache")
    env["XDG_DATA_HOME"] = str(ghidra_user_home / ".local" / "share")
    java_path = shutil.which("java")
    if java_path and "JAVA_HOME" not in env:
        env["JAVA_HOME"] = str(Path(java_path).resolve().parents[1])
    java_options = env.get("JAVA_TOOL_OPTIONS", "")
    localhost_options = "-Djava.net.preferIPv4Stack=true -Djava.rmi.server.hostname=localhost"
    env["JAVA_TOOL_OPTIONS"] = f"{java_options} {localhost_options}".strip()

    result = subprocess.run(
        [
            str(analyze_headless),
            str(project_dir),
            "DffiGhidraTest",
            "-import",
            str(binary_path),
            "-scriptPath",
            str(GHIDRA_SCRIPT_DIR),
            "-postScript",
            "Ghidra2ISF.java",
            str(isf_path),
            "-deleteProject",
        ],
        text=True,
        capture_output=True,
        timeout=180,
        env=env,
    )
    if result.returncode != 0:
        output = result.stdout + result.stderr
        if "InetAddress.getLocalHost" in output or "Name or service not known" in output:
            pytest.skip("Ghidra cannot resolve the container hostname in this environment")
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )

    exported = json.loads(isf_path.read_text(encoding="utf-8"))
    assert exported["metadata"]["producer"]["name"] == "ghidra2isf"
    assert exported["user_types"]
    assert exported["symbols"]
    assert exported["functions"]

    parsed = VtypeJson(str(isf_path))
    assert parsed.get_function("add_packet") is not None


@pytest.mark.skipif(
    os.environ.get("DFFI_GHIDRA_TEST") != "1",
    reason="set DFFI_GHIDRA_TEST=1 to compile bundled Ghidra scripts",
)
def test_bundled_ghidra_scripts_compile_against_ghidra(tmp_path: Path) -> None:
    ghidra_home = _ghidra_home()
    if ghidra_home is None:
        pytest.skip("GHIDRA_HOME/analyzeHeadless not found; set DFFI_GHIDRA_DOWNLOAD=1 to download")
    javac = shutil.which("javac")
    if javac is None:
        pytest.skip("javac is required to compile bundled Ghidra scripts")

    jars = [str(path) for path in ghidra_home.rglob("*.jar")]
    if not jars:
        pytest.skip("No Ghidra jars found")

    classes_dir = tmp_path / "classes"
    classes_dir.mkdir()
    subprocess.run(
        [
            javac,
            "-proc:none",
            "-cp",
            os.pathsep.join(jars),
            "-d",
            str(classes_dir),
            str(EXPORT_SCRIPT_PATH),
            str(IMPORT_SCRIPT_PATH),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
