"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu import picmi

import unittest


class TestSpecies(unittest.TestCase):
    def setUp(self):
        grid = picmi.Cartesian3DGrid(number_of_cells=[192, 2048, 12],
                                     lower_bound=[0, 0, 0],
                                     upper_bound=[
                                         3.40992e-5, 9.07264e-5, 2.1312e-6],
                                     lower_boundary_conditions=[
                                         "open", "open", "periodic"],
                                     upper_boundary_conditions=[
                                         "open", "open", "periodic"])
        solver = picmi.ElectromagneticSolver(method='Yee', grid=grid)
        sim = picmi.Simulation(time_step_size=1.39e-16,
                               max_steps=int(2048),
                               solver=solver)

        laser = picmi.GaussianLaser(0.8e-6, 5.0e-6 / 1.17741, 5.0e-15,
                                    a0=8,
                                    propagation_direction=[0, 1, 0],
                                    polarization_direction=[1, 0, 0],
                                    centroid_position=[
                                        0.5*grid.upper_bound[0],
                                        0,
                                        0.5*grid.upper_bound[2]],
                                    focal_position=[0.5*grid.upper_bound[0],
                                                    4.62e-5,
                                                    0.5*grid.upper_bound[2]])
        sim.add_laser(laser, None)

        self.laser = laser
        self.grid = grid
        self.solver = solver
        self.sim = sim

    def test_hydrogen_atoms(self):
        """create hydrogen atoms as simple as possible"""
        uniform_dist = picmi.UniformDistribution(density=8e24)
        species_hydrogen = picmi.Species(name="hydrogen", particle_type="H",
                                         charge_state=0,
                                         initial_distribution=uniform_dist,
                                         density_scale=3)
        random_layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)

        self.sim.add_species(species_hydrogen, random_layout)
        self.sim.picongpu_run()
