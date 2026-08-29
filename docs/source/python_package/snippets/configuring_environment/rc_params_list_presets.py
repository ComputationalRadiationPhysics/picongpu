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

from picongpu import core

for entry in sorted((core.path("etc") / "picongpu").iterdir()):
    if entry.is_dir() or entry.name.endswith(".profile.example"):
        print(entry.name)
