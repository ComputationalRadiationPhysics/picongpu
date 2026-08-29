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

from picongpu import rc_params

rc_params["preset"] = "rosi-hzdr"

for key, value in rc_params.items():
    print(f"{key}: {value}")

# changing a default
rc_params["tbg_partition"] = "a100"
