
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
            
            # Use cp310-abi3 to satisfy PyPI's strict platform wheel requirements.
            # This ensures pip prefers this wheel over the generic -any.whl, 
            # while keeping it compatible with all Python 3.10+ versions.
            build_data["tag"] = f"cp310-abi3-{plat_tag}"