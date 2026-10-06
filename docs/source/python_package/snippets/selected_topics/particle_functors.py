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
from picongpu.picmi.diagnostics import BinSpec, Binning, BinningAxis, DerivedFieldDump, EnergyHistogram
from picongpu.picmi.particle_functor import (
    FilteredSpecies,
    MacroParticle,
    ParticleFilter,
    ParticleFunctor,
    PhysicalParticle,
)
from picongpu.picmi.particle_functor.unit_dimension import M, UnitDimension

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
def gamma(particle: MacroParticle) -> float:
    mass = particle.get("mass")
    px, py, pz = particle.get("momentum")
    return sqrt(mass**2 + px**2 + py**2 + pz**2) / mass


# END-PARTICLE-FUNCTOR


# BEGIN-PHYSICAL-PARTICLE
# The argument's type annotation declares which particle flavour the functor
# refers to: MacroParticle (the default) or PhysicalParticle.
@ParticleFunctor(unit_dimension=M)
def macroparticle_mass(particle: MacroParticle) -> float:
    return particle.get("mass")


# PhysicalParticle reads the same macro-particle attribute, but the returned
# quantity is interpreted as a single-particle (per-particle) value: the
# weighting factor is symbolically divided out of the scaling-sensitive
# symbols (mass, charge, kinetic energy) at code generation.
@ParticleFunctor(unit_dimension=M)
def physical_mass(particle: PhysicalParticle) -> float:
    return particle.get("mass")


# END-PHYSICAL-PARTICLE


# BEGIN-UNIT-FACTOR
# unit_factor pins the numeric scale returned by getUnit() when the automatic
# derivation cannot handle the unit (here a count whose unit carries N_ppm).
# It takes a number, a sympy expression, or a callable returning one.
@ParticleFunctor(unit_dimension=UnitDimension(N=1), unit_factor=1.0e6)
def explicit_scale(particle: MacroParticle) -> float:
    return 1.0


# END-UNIT-FACTOR


@ParticleFunctor
def count(particle: MacroParticle):
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
def fast(particle: MacroParticle) -> bool:
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

# Derived fields expose the single-particle semantics in the output: the
# physical-particle mass is dumped per particle, while the macro-particle mass
# stays a weighting-scaled grid quantity.
physical_mass_dump = DerivedFieldDump(species=electrons, functor=physical_mass, period=picmi.diagnostics.TS[::10])
macroparticle_mass_dump = DerivedFieldDump(
    species=electrons, functor=macroparticle_mass, period=picmi.diagnostics.TS[::10]
)
# unit_factor pins the numeric getUnit() scale (here 1e6) instead of the
# auto-derived sim.unit.* monomial.
explicit_scale_dump = DerivedFieldDump(species=electrons, functor=explicit_scale, period=picmi.diagnostics.TS[::10])

simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    species=[electrons],
    layouts=[picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)],
    diagnostics=[gamma_distribution, histogram, physical_mass_dump, macroparticle_mass_dump, explicit_scale_dump],
)

simulation.write_input_file(Path("particle_functors_setup"))
