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

Defines a particle functor for the Lorentz factor in a binning diagnostic,
and a particle filter that selects the ultra-relativistic electrons.
"""

from pathlib import Path
from sympy import sqrt

from picongpu import picmi
from picongpu.picmi.diagnostics import BinSpec, Binning, BinningAxis, EnergyHistogram
from picongpu.picmi.particle_functor import FilteredSpecies, ParticleFilter, ParticleFunctor

grid = picmi.Cartesian3DGrid(
    number_of_cells=[32, 32, 32],
    lower_bound=[0, 0, 0],
    upper_bound=[1e-6, 1e-6, 1e-6],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid)

electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=picmi.UniformDistribution(density=1e23, rms_velocity=[0.1 * picmi.constants.c] * 3),
)


# BEGIN-PARTICLE-FUNCTOR
@ParticleFunctor
def gamma(particle):
    mass = particle.get("mass")
    px, py, pz = particle.get("momentum")
    return sqrt(mass**2 + px**2 + py**2 + pz**2) / mass


# END-PARTICLE-FUNCTOR


@ParticleFunctor
def count(particle):
    return 1.0


# a functor used as an axis of a binning diagnostic:
gamma_distribution = Binning(
    name="gammaDistribution",
    deposition_functor=count,
    axes=[BinningAxis(functor=gamma, bin_spec=BinSpec(kind="linear", start=1.0, stop=100.0, nsteps=100))],
    species=electrons,
    period=picmi.diagnostics.TS[::10],
)


# BEGIN-PARTICLE-FILTER
@ParticleFilter
def fast(particle):
    return particle.get("gamma") > 10.0


# a filter wrapped into a species usable wherever a species is accepted;
# its name in the output is "electrons_fast":
fast_electrons = FilteredSpecies(species=electrons, functor=fast)
# END-PARTICLE-FILTER

histogram = EnergyHistogram(
    species=fast_electrons,
    period=picmi.diagnostics.TS[-1],
    bin_count=100,
    min_energy=0.0,
    max_energy=1000.0 * picmi.constants.keV,
)

simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    species=[electrons],
    layouts=[picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)],
    diagnostics=[gamma_distribution, histogram],
)

simulation.write_input_file(Path("particle_functors_setup"))
