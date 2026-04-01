# hatch_build.py
import os
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        # Look for our custom CI environment variable
        plat_tag = os.environ.get("DWARFFI_PLATFORM_TAG")
        
        if plat_tag:
            # If set, force the wheel to be platform-specific instead of 'any'
            build_data["pure_python"] = False
            build_data["tag"] = f"py3-none-{plat_tag}"