#!/usr/bin/env python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "numpy",
#   "scipy",
#   "sympy",
#   "picongpu @ git+https://github.com/chillenzer/picongpu@add-env-management-to-python-package#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Masoud Afshari, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import sympy
from picongpu import picmi
from picongpu.picmi.diagnostics import binning
from picongpu.picmi.diagnostics.radiation import RadiationObserverConfiguration
from picongpu.picmi.particle_functor.unit_dimension import I, L, M, T
from scipy.constants import c, elementary_charge

"""
@file PICMI user script reproducing the PIConGPU LWFA example

This Python script is example PICMI user script reproducing the LaserWakefield example setup, based on 8.cfg.

"""

# generation modifiers
ENABLE_IONS = True
ENABLE_IONIZATION = True
ADD_CUSTOM_INPUT = True
OUTPUT_DIRECTORY_PATH = Path("lwfa")
MODE: Literal["run", "write"] = "run"


numberCells = np.array([192, 2048, 192])
cellSize = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])  # unit: meter

# Define the simulation grid based on grid.param
grid = picmi.Cartesian3DGrid(
    picongpu_n_gpus=[2, 4, 1],
    number_of_cells=numberCells.tolist(),
    lower_bound=[0, 0, 0],
    upper_bound=(numberCells * cellSize).tolist(),
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)

gaussianProfile = picmi.distribution.GaussianDistribution(
    density=1.0e25,
    center_front=8.0e-5,
    sigma_front=8.0e-5,
    center_rear=10.0e-5,
    sigma_rear=8.0e-5,
    factor=-1.0,
    power=4.0,
    vacuum_front=50 * cellSize[1],
)

solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")

laser_duration = 5.0e-15
pulse_init = 15.0
laser = picmi.GaussianLaser(
    wavelength=0.8e-6,
    waist=5.0e-6 / 1.17741,
    duration=laser_duration,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    focal_position=[
        float(numberCells[0] * cellSize[0] / 2.0),
        4.62e-5,
        float(numberCells[2] * cellSize[2] / 2.0),
    ],
    centroid_position=[
        float(numberCells[0] * cellSize[0] / 2.0),
        -0.5 * pulse_init * laser_duration * c,
        float(numberCells[2] * cellSize[2] / 2.0),
    ],
    picongpu_polarization_type=picmi.lasers.PolarizationType.LINEAR,
    a0=8.0,
    phi0=0.0,
)

random_layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)

# Initialize particles  based on speciesInitialization.param
# simulation schema : https://github.com/BrianMarre/picongpu/blob/2ddcdab4c1aca70e1fc0ba02dbda8bd5e29d98eb/share/picongpu/pypicongpu/schema/simulation.Simulation.json

# for particle type see https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_SpeciesType.md
species_list = []
if not ENABLE_IONIZATION:
    interaction = []

    electrons = picmi.Species(particle_type="electron", name="electron", initial_distribution=gaussianProfile)
    species_list.append((electrons, random_layout))

    if ENABLE_IONS:
        hydrogen_fully_ionized = picmi.Species(
            particle_type="H",
            name="hydrogen",
            picongpu_fixed_charge=True,
            initial_distribution=gaussianProfile,
        )
        species_list.append((hydrogen_fully_ionized, random_layout))
else:
    if not ENABLE_IONS:
        raise ValueError("Ions species required for ionization")

    hydrogen_with_ionization = picmi.Species(
        particle_type="H", name="hydrogen", charge_state=0, initial_distribution=gaussianProfile
    )
    species_list.append((hydrogen_with_ionization, random_layout))

    electrons = picmi.Species(particle_type="electron", name="electron", initial_distribution=None)
    species_list.append((electrons, None))

    adk_ionization_model = picmi.ADK(
        ADK_variant=picmi.ADKVariant.LinearPolarization,
        ion_species=hydrogen_with_ionization,
        ionization_electron_species=electrons,
        ionization_current=None,
    )

    bsi_effectiveZ_ionization_model = picmi.BSI(
        BSI_extensions=[picmi.BSIExtension.EffectiveZ],
        ion_species=hydrogen_with_ionization,
        ionization_electron_species=electrons,
        ionization_current=None,
    )
    interaction = [adk_ionization_model, bsi_effectiveZ_ionization_model]
sim = picmi.Simulation(
    solver=solver,
    max_steps=4000,
    time_step_size=1.39e-16,
    picongpu_moving_window_move_point=0.9,
    picongpu_walltime=datetime.timedelta(hours=2.0),
    picongpu_interaction=interaction,
)

for species, layout in species_list:
    sim.add_species(species, layout=layout)


# define e-spec binning plugin
def computeEnergy(particle):
    return particle.get("kinetic energy")


def computeAngle(particle):
    px, py, pz = particle.get("momentum")
    return sympy.atan2(px, py)


def computeCharge(particle):
    return particle.get("charge")


energyFunctor = binning.BinningFunctor(
    name="energy", functor=computeEnergy, return_type=float, unit_dimension=L**2 * M * T**-2
)
thetaFunctor = binning.BinningFunctor(
    name="theta",
    functor=computeAngle,
    return_type=float,
)

maxEnergy_MeV = 100.0
energyRange = binning.BinSpec("linear", 0.0, maxEnergy_MeV * 1e6 * elementary_charge, 800)  # convert MeV to Joule
thetaRange = binning.BinSpec("linear", -0.250, +0.250, 256)  # in rad

energyAxis = binning.BinningAxis(functor=energyFunctor, bin_spec=energyRange, name="energy")
thetaAxis = binning.BinningAxis(functor=thetaFunctor, bin_spec=thetaRange, name="theta")

eSpec_deposition_functor = binning.BinningFunctor(
    name="eSpec", functor=computeCharge, return_type=float, unit_dimension=I * T
)

eSPec_binning = binning.Binning(
    name="eSpec",
    deposition_functor=eSpec_deposition_functor,
    axes=[energyAxis, thetaAxis],
    species=electrons,
    period=picmi.diagnostics.TimeStepSpec[::100],
)


N_OBSERVER = 256
sim.diagnostics = [
    eSPec_binning,
    picmi.diagnostics.PhaseSpace(
        species=electrons,
        # Resulting values for period:
        # 0, 17, 50, 57, 64, 71, 100, 200, ...
        period=picmi.diagnostics.TimeStepSpec[::100, 50:72:7, 17],
        spatial_coordinate="y",
        momentum_coordinate="py",
        min_momentum=-1.0,
        max_momentum=1.0,
    ),
    picmi.diagnostics.EnergyHistogram(
        species=electrons,
        # Resulting values for period:
        # 0, 100, 200, ...
        period=picmi.diagnostics.TimeStepSpec[::100],
        bin_count=1024,
        min_energy=0.0,
        max_energy=1000.0,
    ),
    picmi.diagnostics.MacroParticleCount(
        species=electrons,
        period=picmi.diagnostics.TimeStepSpec[::100],
    ),
    picmi.diagnostics.Checkpoint(
        period=picmi.diagnostics.TimeStepSpec[::100],
    ),
    picmi.diagnostics.Radiation(
        species=electrons,
        period=picmi.diagnostics.TimeStepSpec[100::100],
        observer=RadiationObserverConfiguration(
            N_observer=N_OBSERVER,
            index_to_direction=lambda i: [
                sympy.sin(2 * sympy.pi / N_OBSERVER * i),
                sympy.cos(2 * sympy.pi / N_OBSERVER * i),
                0,
            ],
        ),
    ),
]

sim.add_laser(laser, None)

if __name__ == "__main__":
    match MODE:
        case "run":
            sim.run(setup_dir=OUTPUT_DIRECTORY_PATH / "setup", run_dir=OUTPUT_DIRECTORY_PATH / "run")
        case "write":
            sim.write_input_file(OUTPUT_DIRECTORY_PATH / "setup")
        case _:
            raise ValueError(f"Unknown {MODE=}.")
