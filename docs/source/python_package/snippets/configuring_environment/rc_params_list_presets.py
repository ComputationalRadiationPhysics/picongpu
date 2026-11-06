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
Authors: Julian Lenz
License: GPLv3+
"""

# BEGIN-RC-LIST-PRESETS
from picongpu import core
from picongpu._rc_params import get_preset_name

# list the names of the presets that can be given to rc_params["preset"]:
# a system directory with a single profile example is a preset by itself;
# a directory with several examples contributes canonical names such as
# "bash/spack" or "hemera-hzdr/gpu".
root = core.path("etc") / "picongpu"
for entry in sorted(root.iterdir()):
    if entry.is_file():
        if entry.name.endswith(".profile.example"):
            print(get_preset_name(entry.name))
        continue
    profiles = sorted(entry.glob("*.profile.example"))
    if len(profiles) == 1:
        print(entry.name)
    else:
        for profile in profiles:
            print(get_preset_name(f"{entry.name}/{profile.name}"))
# END-RC-LIST-PRESETS
