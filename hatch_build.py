
# hatch_build.py
import os

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Look for our custom CI environment variable
        plat_tag = os.environ.get("DWARFFI_PLATFORM_TAG")
        
        if plat_tag:
            # We must explicitly override all three parts of the tag (python, abi, platform)
            # Otherwise, setting pure_python=False forces the current CPython ABI tag.
            build_data["pure_python"] = False
            build_data["infer_tag"] = False
            
            # This explicitly formats the filename and internal metadata to: py3-none-<platform>
            build_data["tag"] = f"py3-none-{plat_tag}"