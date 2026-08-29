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

import json
from pathlib import Path

from picongpu import picmi

grid = picmi.Cartesian3DGrid(
    number_of_cells=[192, 2048, 192],
    lower_bound=[0, 0, 0],
    upper_bound=[0.1772e-6, 0.4430e-7, 0.1772e-6],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid)
simulation = picmi.Simulation(max_steps=100, solver=solver)

# serialize the PyPIConGPU representation of the simulation
pypicongpu_simulation = simulation.get_as_pypicongpu()
simulation_json = pypicongpu_simulation.model_dump(mode="json")
print(f"serialized simulation into {len(simulation_json)} top-level fields")
# the same JSON representation is written to metadata/pypicongpu_runner.json
# upon generation of the input files

# individual elements such as Species are Pydantic models as well:


def serialize_species(species, path):
    with Path(path).open("w") as file:
        json.dump(species.model_dump(mode="json"), file)


def deserialize_species(path):
    with Path(path).open("r") as file:
        return picmi.Species.model_validate(json.load(file))


electrons = picmi.Species(name="electrons", particle_type="electron")
serialize_species(electrons, "electrons.json")
electrons_round_trip = deserialize_species("electrons.json")
assert electrons_round_trip.model_dump() == electrons.model_dump()
print("It worked!")
