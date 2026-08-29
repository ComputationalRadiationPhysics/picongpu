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

# may raise an exception when any non-default configuration was applied already:
rc_params["preset"] = "bash"

# temporarily disable that exception:
with rc_params.set_temporarily(dirty_reset_policy="ignore"):
    rc_params["preset"] = "bash"

# permanently disable that exception:
rc_params["dirty_reset_policy"] = "warn"
rc_params["preset"] = "bash"
