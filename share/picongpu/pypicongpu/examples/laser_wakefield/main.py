"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Masoud Afshari, Brian Edward Marre
License: GPLv3+
"""

from picongpu import picmi
from picongpu import pypicongpu
import numpy as np

"""
@file PICMI user script reproducing the PIConGPU LWFA example

This Python script is example PICMI user script reproducing the LaserWakefield example setup, based on 8.cfg.
"""


# generation modifiers
ENABLE_IONS = True
ENABLE_IONIZATION = True
ADD_CUSTOM_INPUT = True
OUTPUT_DIRECTORY_PATH = "LWFA"

numberCells = np.array([192, 2048, 192])
cellSize = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])  # unit: meter)

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
    vacuum_cells_front=50,
)

# for particle type see https://github.com/openPMD/openPMD-standard/blob/upcoming-2.0.0/EXT_SpeciesType.md
electrons = picmi.Species(particle_type="electron", name="electron", initial_distribution=gaussianProfile)

hydrogen_ionization = picmi.Species(
    particle_type="H", name="hydrogen", charge_state=0, initial_distribution=gaussianProfile
)

hydrogen_fully_ionized = picmi.Species(
    particle_type="H", name="hydrogen", picongpu_fixed_charge=True, initial_distribution=gaussianProfile
)

solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="Yee",
)

laser = picmi.GaussianLaser(
    wavelength=0.8e-6,
    waist=5.0e-6 / 1.17741,
    duration=5.0e-15,
    propagation_direction=[0.0, 1.0, 0.0],
    polarization_direction=[1.0, 0.0, 0.0],
    focal_position=[float(numberCells[0] * cellSize[0] / 2.0), 4.62e-5, float(numberCells[2] * cellSize[2] / 2.0)],
    centroid_position=[float(numberCells[0] * cellSize[0] / 2.0), 0.0, float(numberCells[2] * cellSize[2] / 2.0)],
    picongpu_polarization_type=pypicongpu.laser.GaussianLaser.PolarizationType.CIRCULAR,
    a0=8.0,
    picongpu_phase=0.0,
)

randomLayout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)

# Initialize particles  based on speciesInitialization.param
# simulation schema : https://github.com/BrianMarre/picongpu/blob/2ddcdab4c1aca70e1fc0ba02dbda8bd5e29d98eb/share/picongpu/pypicongpu/schema/simulation.Simulation.json

if not ENABLE_IONIZATION:
    hydrogen = hydrogen_fully_ionized
    interaction = None
else:
    hydrogen = hydrogen_ionization
    adk_ionization_model = picmi.ADK(
        ADK_variant=picmi.ADKVariant.CircularPolarization,
        ion_species=hydrogen_ionization,
        ionization_electron_species=electrons,
        ionization_current=None,
    )

    bsi_effectiveZ_ionization_model = picmi.BSI(
        BSI_extensions=[picmi.BSIExtension.EffectiveZ],
        ion_species=hydrogen_ionization,
        ionization_electron_species=electrons,
        ionization_current=None,
    )

    interaction = picmi.Interaction(
        ground_state_ionization_model_list=[adk_ionization_model, bsi_effectiveZ_ionization_model]
    )

sim = picmi.Simulation(
    solver=solver,
    max_steps=4000,
    time_step_size=1.39e-16,
    picongpu_moving_window_move_point=0.9,
    picongpu_interaction=interaction,
)

sim.add_species(electrons, layout=randomLayout)

if ENABLE_IONS:
    sim.add_species(hydrogen, layout=randomLayout)

sim.add_laser(laser, None)

# additional non standardized custom user input
# only active if custom templates are used

# for generating setup with custom input see standard implementation,
#  see https://picongpu.readthedocs.io/en/latest/usage/picmi/custom_template.html
if ADD_CUSTOM_INPUT:
    min_weight_input = pypicongpu.customuserinput.CustomUserInput()
    min_weight_input.addToCustomInput({"minimum_weight": 10.0}, "minimum_weight")
    sim.picongpu_add_custom_user_input(min_weight_input)

    output_configuration = pypicongpu.customuserinput.CustomUserInput()
    output_configuration.addToCustomInput(
        {
            "png_plugin_data_list": "['Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz', 'Jx', 'Jy', 'Jz']",
            "png_plugin_SCALE_IMAGE": 1.0,
            "png_plugin_SCALE_TO_CELLSIZE": True,
            "png_plugin_WHITE_BOX_PER_GPU": False,
            "png_plugin_EM_FIELD_SCALE_CHANNEL1": 7,
            "png_plugin_EM_FIELD_SCALE_CHANNEL2": -1,
            "png_plugin_EM_FIELD_SCALE_CHANNEL3": -1,
            "png_plugin_CUSTOM_NORMALIZATION_SI": "5.0e12 / constants.c, 5.0e12, 15.0",
            "png_plugin_PRE_PARTICLE_DENS_OPACITY": 0.25,
            "png_plugin_PRE_CHANNEL1_OPACITY": 1.0,
            "png_plugin_PRE_CHANNEL2_OPACITY": 1.0,
            "png_plugin_PRE_CHANNEL3_OPACITY": 1.0,
            "png_plugin_preParticleDensCol": "colorScales::grayInv",
            "png_plugin_preChannel1Col": "colorScales::green",
            "png_plugin_preChannel2Col": "colorScales::none",
            "png_plugin_preChannel3Col": "colorScales::none",
            "png_plugin_preChannel1": "field_E.x() * field_E.x();",
            "png_plugin_preChannel2": "field_E.y()",
            "png_plugin_preChannel3": "-1.0_X * field_E.y()",
            "png_plugin_period": 100,
            "png_plugin_axis": "yx",
            "png_plugin_slicePoint": 0.5,
            "png_plugin_species_name": "electron",
            "png_plugin_folder_name": "pngElectronsYX",
        },
        "png plugin configuration",
    )

    output_configuration.addToCustomInput(
        {
            "energy_histogram_species_name": "electron",
            "energy_histogram_period": 100,
            "energy_histogram_bin_count": 1024,
            "energy_histogram_min_energy": 0.0,
            "energy_histogram_maxEnergy": 1000.0,
            "energy_histogram_filter": "all",
        },
        "energy histogram plugin configuration",
    )

    output_configuration.addToCustomInput(
        {
            "phase_space_species_name": "electron",
            "phase_space_period": 100,
            "phase_space_space": "y",
            "phase_space_momentum": "py",
            "phase_space_min": -1.0,
            "phase_space_max": 1.0,
            "phase_space_filter": "all",
        },
        "phase space plugin configuration",
    )

    output_configuration.addToCustomInput(
        {"opnePMD_period": 100, "opnePMD_file": "simData", "opnePMD_extension": "bp"}, "openPMD plugin configuration"
    )

    output_configuration.addToCustomInput(
        {"checkpoint_period": 100, "checkpoint_backend": "openPMD", "checkpoint_restart_backend": "openPMD"},
        "checkpoint configuration",
    )

    output_configuration.addToCustomInput(
        {"macro_particle_count_period": 100, "macro_particle_count_species_name": "electron"},
        "macro particle count plugin configuration",
    )
    sim.picongpu_add_custom_user_input(output_configuration)

sim.write_input_file(OUTPUT_DIRECTORY_PATH)
