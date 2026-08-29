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

Defines three small simulations demonstrating the supported interactions:
ADK tunnel ionization of hydrogen,
BSI (with the Stark-shift extension) ionization of hydrogen,
and synchrotron radiation from electrons.
"""

from pathlib import Path

from picongpu import picmi

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

# BEGIN-INTERACTIONS-ADK
hydrogen = picmi.Species(
    name="hydrogen",
    particle_type="H",
    charge_state=0,
    initial_distribution=distribution,
)
electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=None,
)

adk = picmi.ADK(
    ADK_variant=picmi.ADKVariant.LinearPolarization,
    ion_species=hydrogen,
    ionization_electron_species=electrons,
    ionization_current=None,
)

sim_adk = picmi.Simulation(max_steps=10, solver=solver, picongpu_interaction=[adk])
sim_adk.add_species(hydrogen, layout)
sim_adk.add_species(electrons, None)
sim_adk.run(setup_dir=Path("adk_setup"), run_dir=Path("adk_run"))
# END-INTERACTIONS-ADK

# BEGIN-INTERACTIONS-BSI
bsi_hydrogen = picmi.Species(
    name="bsi_hydrogen",
    particle_type="H",
    charge_state=0,
    initial_distribution=distribution,
)
bsi_electrons = picmi.Species(
    name="bsi_electrons",
    particle_type="electron",
    initial_distribution=None,
)

bsi = picmi.BSI(
    ion_species=bsi_hydrogen,
    ionization_electron_species=bsi_electrons,
    ionization_current=None,
    BSI_extensions=(picmi.BSIExtension.StarkShift,),
)

sim_bsi = picmi.Simulation(max_steps=10, solver=solver, picongpu_interaction=[bsi])
sim_bsi.add_species(bsi_hydrogen, layout)
sim_bsi.add_species(bsi_electrons, None)
sim_bsi.run(setup_dir=Path("bsi_setup"), run_dir=Path("bsi_run"))
# END-INTERACTIONS-BSI

# BEGIN-INTERACTIONS-SYNCHROTRON
sync_electrons = picmi.Species(
    name="sync_electrons",
    particle_type="electron",
    initial_distribution=distribution,
)
photons = picmi.Species(
    name="photons",
    particle_type="photon",
    initial_distribution=None,
)

synchrotron = picmi.Synchrotron(electron_species=sync_electrons, photon_species=photons)

sim_sync = picmi.Simulation(max_steps=10, solver=solver, picongpu_interaction=[synchrotron])
sim_sync.add_species(sync_electrons, layout)
sim_sync.add_species(photons, None)
sim_sync.run(setup_dir=Path("synchrotron_setup"), run_dir=Path("synchrotron_run"))
# END-INTERACTIONS-SYNCHROTRON
