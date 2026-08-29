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

Demonstrates the TimeStepSpec syntax used by all diagnostics:
slices and indices in units of simulation steps or of physical time,
and how specifications in different units can be combined.
"""

from picongpu.picmi.diagnostics import TimeStepSpec

# every 10th step: 0, 10, 20, ...
periodic = TimeStepSpec[::10]("steps")
print("periodic specs:", periodic.specs)

# the first 5 steps (0 to 5 inclusively) and every step from step 49 to the end
head_and_tail = TimeStepSpec[:5, 49:]("steps")
print("head and tail specs:", head_and_tail.specs)

# every 200 femtoseconds between 1 and 5 attoseconds
physical_time = TimeStepSpec[1.0e-15:5.0e-15:2.0e-16]("seconds")
print("physical time specs:", physical_time.specs_in_seconds)

# specifications in different units can be combined (set union)
combined = periodic + physical_time
print("combined unit system:", combined.unit_system)

print("It worked!")
