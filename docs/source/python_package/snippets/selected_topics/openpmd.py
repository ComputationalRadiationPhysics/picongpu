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

Defines a simulation with openPMD-based output:
a particle dump of the electrons, a dump of the native electric field
and a dump of a derived field (the electron kinetic energy),
all every 5th step.
Diagnostics that share the same output configuration
(both default to a file named "simData")
end up in the same openPMD file.
The magnetic field is written to a separate file,
which results in a second openPMD configuration.
"""

from pathlib import Path

from picongpu import picmi
from picongpu.picmi.diagnostics import (
    DerivedFieldDump,
    NativeFieldDump,
    OpenPMDConfig,
    ParticleDump,
    TimeStepSpec,
)
from picongpu.picmi.particle_functor import ParticleFunctor


def kinetic_energy(particle):
    return particle.get("kinetic energy")


grid = picmi.Cartesian3DGrid(
    number_of_cells=[32, 32, 32],
    lower_bound=[0, 0, 0],
    upper_bound=[1e-6, 1e-6, 1e-6],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid)
distribution = picmi.UniformDistribution(density=1e23)
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=1)
electrons = picmi.Species(name="electrons", particle_type="electron", initial_distribution=distribution)

sim = picmi.Simulation(max_steps=100, solver=solver)
sim.add_species(electrons, layout)

sim.add_diagnostic(ParticleDump(species=electrons, period=TimeStepSpec[::5]))
sim.add_diagnostic(NativeFieldDump(fieldname="E", period=TimeStepSpec[::5]))
sim.add_diagnostic(
    DerivedFieldDump(
        species=electrons,
        functor=ParticleFunctor(functor=kinetic_energy, name="kineticEnergy"),
        period=TimeStepSpec[::5],
    )
)
sim.add_diagnostic(
    NativeFieldDump(
        fieldname="B",
        period=TimeStepSpec[::5],
        options=OpenPMDConfig(file="magneticField"),
    )
)

sim.run(setup_dir=Path("openpmd_setup"), run_dir=Path("openpmd_run"))

for config in sorted(Path("openpmd_setup").joinpath("etc").glob("openPMD_config_*.toml")):
    print(config.name)
    print(config.read_text())
