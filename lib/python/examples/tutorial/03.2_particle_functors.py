#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "numpy",
#   "picongpu @ git+https://github.com/chillenzer/picongpu@add-env-management-to-python-package#subdirectory=lib/python"
# ]
# ///

from pathlib import Path

import numpy as np
from picongpu.picmi import (
    ADK,
    ADKVariant,
    Cartesian3DGrid,
    ElectromagneticSolver,
    PseudoRandomLayout,
    Simulation,
    Species,
)
from picongpu.picmi.constants import c
from picongpu.picmi.diagnostics import Checkpoint, MacroParticleCount, TimeStepSpec
from picongpu.picmi.diagnostics.binning import Binning, BinningAxis, BinSpec
from picongpu.picmi.diagnostics.field_dump import DerivedFieldDump
from picongpu.picmi.diagnostics.particle_dump import ParticleDump
from picongpu.picmi.distribution import GaussianDistribution
from picongpu.picmi.lasers import GaussianLaser, PolarizationType
from picongpu.picmi.particle_functor.particle_filter import FilteredSpecies, ParticleFilter
from picongpu.picmi.particle_functor.particle_functor import ParticleFunctor
from picongpu.picmi.particle_functor.rng_arg import RNGArg
from picongpu.picmi.particle_functor.unit_dimension import L, M, T

NUM_CELLS = np.array([192, 2048, 192])
CELL_SIZE = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])

grid = Cartesian3DGrid(
    number_of_cells=NUM_CELLS,
    lower_bound=[0, 0, 0],
    upper_bound=NUM_CELLS * CELL_SIZE,
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = ElectromagneticSolver(method="Yee", grid=grid, cfl=1.0)

LASER_DURATION = 5.0e-15
PULSE_INIT = 15.0

laser = GaussianLaser(
    wavelength=0.8e-6,
    waist=5.0e-6 / 1.17741,
    duration=LASER_DURATION,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    focal_position=[
        float(NUM_CELLS[0] * CELL_SIZE[0] / 2.0),
        4.62e-5,
        float(NUM_CELLS[2] * CELL_SIZE[2] / 2.0),
    ],
    centroid_position=[
        float(NUM_CELLS[0] * CELL_SIZE[0] / 2.0),
        -0.5 * PULSE_INIT * LASER_DURATION * c,
        float(NUM_CELLS[2] * CELL_SIZE[2] / 2.0),
    ],
    picongpu_polarization_type=PolarizationType.LINEAR,
    a0=8.0,
    phi0=0.0,
)

particle_distribution = GaussianDistribution(
    density=1.0e25,
    center_front=8.0e-5,
    sigma_front=8.0e-5,
    center_rear=10.0e-5,
    sigma_rear=8.0e-5,
    factor=-1.0,
    power=4.0,
    vacuum_front=50 * CELL_SIZE[1],
)
particle_layout = PseudoRandomLayout(n_macroparticles_per_cell=2)

electrons = Species(particle_type="electron", name="electrons", initial_distribution=particle_distribution)
hydrogen_ions = Species(
    particle_type="H",
    name="hydrogen",
    charge_state=0,
    initial_distribution=particle_distribution,
)

adk_ionization = ADK(
    ADK_variant=ADKVariant.LinearPolarization,
    ion_species=hydrogen_ions,
    ionization_electron_species=electrons,
    ionization_current=None,
)

checkpoint = Checkpoint(period=TimeStepSpec[::100])
macro_particle_count = MacroParticleCount(
    species=electrons,
    # Resulting values for period:
    # 0, 17, 50, 57, 64, 71, 100, 200, ...
    period=TimeStepSpec[::100, 50:72:7, 17],
)


@ParticleFunctor
def kinetic_energy_density(macro_particle):
    return macro_particle.get("kinetic energy") / np.prod(CELL_SIZE)


electron_energy_density = DerivedFieldDump(
    species=electrons, functor=kinetic_energy_density, period=TimeStepSpec[::100]
)


@ParticleFunctor(unit_dimension=M * L / T)
def momentum_x(macro_particle):
    return macro_particle.get("momentum")[0]


@ParticleFunctor(unit_dimension=M * L / T)
def momentum_y(macro_particle):
    return macro_particle.get("momentum")[1]


@ParticleFunctor(unit_dimension=M * L / T)
def momentum_z(macro_particle):
    return macro_particle.get("momentum")[2]


@ParticleFunctor(return_type=int)
def macro_particles(_):
    return 1


MOMENTUM_MAX = 10.0
momentum_bins = BinSpec(kind="linear", start=-MOMENTUM_MAX, stop=MOMENTUM_MAX, nsteps=100)

electron_momentum = Binning(
    name="electron_momentum",
    species=electrons,
    deposition_functor=macro_particles,
    axes=[BinningAxis(functor=f, bin_spec=momentum_bins) for f in (momentum_x, momentum_y, momentum_z)],
    period=TimeStepSpec[::100],
)


@ParticleFilter
def random_half(_, rng: RNGArg):
    return rng.get("uniform", range=(0.0, 1.0)) < 0.5


random_electrons = ParticleDump(species=FilteredSpecies(species=electrons, functor=random_half))

sim = Simulation(
    max_steps=1000,
    solver=solver,
    picongpu_lasers=laser,
    picongpu_species=[electrons, hydrogen_ions],
    picongpu_particle_layout=particle_layout,
    picongpu_diagnostics=[
        checkpoint,
        macro_particle_count,
        electron_energy_density,
        electron_momentum,
        random_electrons,
    ],
)

OUTPUT_PATH = Path(__file__[: -len(".py")])
sim.run(setup_dir=OUTPUT_PATH / "setup", run_dir=OUTPUT_PATH / "run")
