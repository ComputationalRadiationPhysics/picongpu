#!/usr/bin/env python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: opencode
License: GPLv3+
"""

# BEGIN-RC-LIST-PRESETS
from picongpu import core

# list the names of the presets that can be given to rc_params["preset"]:
# a system directory with a single profile example is a preset by itself,
# a system directory with several profile examples contributes one preset per file
for entry in sorted((core.path("etc") / "picongpu").iterdir()):
    if entry.is_file():
        if entry.name.endswith(".profile.example"):
            print(entry.name)
        continue
    profiles = sorted(entry.glob("*.profile.example"))
    if len(profiles) == 1:
        print(entry.name)
    else:
        for profile in profiles:
            print(f"{entry.name}/{profile.name}")
# END-RC-LIST-PRESETS
