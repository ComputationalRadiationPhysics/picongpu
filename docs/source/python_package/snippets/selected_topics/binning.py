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

Defines a simulation with two binning diagnostics:
a 1D histogram of the electron Lorentz factor gamma
(100 linear bins between 1 and 100), written every 10th step,
once for all electrons and once restricted to a filtered species
(electrons with more than 10 keV kinetic energy).
"""

from pathlib import Path
from sympy import sqrt

from picongpu import picmi
from picongpu.picmi.diagnostics import BinSpec, Binning, BinningAxis, TS
from picongpu.picmi.particle_functor import FilteredSpecies, ParticleFilter, ParticleFunctor


@ParticleFunctor
def gamma(particle):
    mass = particle.get("mass")
    px, py, pz = particle.get("momentum")
    return sqrt(mass**2 + px**2 + py**2 + pz**2) / mass


@ParticleFunctor
def count(particle):
    return 1.0


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

binning = Binning(
    name="gammaDistribution",
    deposition_functor=count,
    axes=[
        BinningAxis(
            functor=gamma,
            bin_spec=BinSpec(kind="linear", start=1.0, stop=100.0, nsteps=100),
        ),
    ],
    species=electrons,
    period=TS[::10],
)


# BEGIN-BINNING-FILTER
@ParticleFilter
def fast(particle):
    # 1.6e-15 J = 10 keV
    return particle.get("kinetic energy") > 1.6e-15


fast_electrons = FilteredSpecies(species=electrons, functor=fast)
# END-BINNING-FILTER

# the same binning, restricted to the filtered species
# (the filtered species name is <species>_<filter> = "electrons_fast")
fast_binning = Binning(
    name="fastGammaDistribution",
    deposition_functor=count,
    axes=[
        BinningAxis(
            functor=gamma,
            bin_spec=BinSpec(kind="linear", start=1.0, stop=100.0, nsteps=100),
        ),
    ],
    species=fast_electrons,
    period=TS[::10],
)

sim = picmi.Simulation(
    max_steps=100,
    solver=solver,
    species=[electrons],
    layouts=[layout],
    diagnostics=[binning, fast_binning],
)

sim.run(setup_dir=Path("binning_setup"), run_dir=Path("binning_run"))
