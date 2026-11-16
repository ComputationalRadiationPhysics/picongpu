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

Shows the public Python import surface: how to import the frontend and
which names are re-exported at which level.
"""

# BEGIN-IMPORT-SURFACE
from picongpu import picmi
from picongpu.picmi import particle_functor, unit_dimension

# flag classes are re-exported at the top level; no deep import needed:
print("polarization types:", [member.name for member in picmi.PolarizationType])
print("collisional setup:", picmi.CollisionalPhysicsSetup.__name__)
print("precision/memory:", picmi.PrecisionConfig.__name__, picmi.MemoryConfig.__name__)

# the functor-authoring helpers live in picmi.particle_functor:
print("rng argument type:", particle_functor.RNGArg.__name__)
energy = unit_dimension.M * unit_dimension.L**2 / unit_dimension.T**2
print("energy unit dimension:", energy)

# diagnostics re-export their helpers and the TS shorthand:
print("TS is TimeStepSpec:", picmi.diagnostics.TS is picmi.diagnostics.TimeStepSpec)
print("radiation observer:", picmi.diagnostics.RadiationObserverConfiguration.__name__)
# END-IMPORT-SURFACE

print("It worked!")
