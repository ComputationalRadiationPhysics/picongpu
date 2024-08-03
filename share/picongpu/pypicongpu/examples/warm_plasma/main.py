"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen
License: GPLv3+
"""

from picongpu import picmi

OUTPUT_DIRECTORY_PATH = "warm_plasma"

boundary_conditions = ["periodic", "periodic", "periodic"]
grid = picmi.Cartesian3DGrid(
    # note: [x] * 3 == [x, x, x]
    number_of_cells=[192] * 3,
    lower_bound=[0, 0, 0],
    upper_bound=[0.0111152256] * 3,
    # delta {x, y, z} is computed implicitly
    # lower & upper boundary conditions must be equal
    lower_boundary_conditions=boundary_conditions,
    upper_boundary_conditions=boundary_conditions,
)
solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)

profile = picmi.UniformDistribution(
    density=1e20,
    # most probable E_kin = 5 mc^2
    # approx. 9000 keV for electrons
    # must be equal for all three components
    rms_velocity=[4.18 * picmi.constants.c] * 3,
)
electron = picmi.Species(
    name="e",
    # openPMD particle type
    particle_type="electron",
    initial_distribution=profile,
)

sim = picmi.Simulation(
    time_step_size=9.65531e-14,
    max_steps=1024,
    solver=solver,
)

layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=25)
sim.add_species(electron, layout)

if __name__ == "__main__":
    sim.write_input_file(OUTPUT_DIRECTORY_PATH)
