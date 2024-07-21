"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Alexander Debus, Richard Pausch
License: GPLv3+
"""

from picongpu.pypicongpu.simulation import Simulation
from picongpu.pypicongpu.laser import GaussianLaser
from picongpu.pypicongpu import grid, solver, species, customuserinput
from picongpu.pypicongpu import rendering

import unittest
import typeguard

from copy import deepcopy


@typeguard.typechecked
def helper_get_species(name: str) -> species.Species:
    spec = species.Species()
    spec.name = name
    spec.constants = []
    spec.attributes = [species.attribute.Position()]
    return spec


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.s = Simulation()
        self.s.delta_t_si = 13.37
        self.s.time_steps = 42
        self.s.typical_ppc = 1
        self.s.grid = grid.Grid3D()
        self.s.grid.cell_size_x_si = 1
        self.s.grid.cell_size_y_si = 2
        self.s.grid.cell_size_z_si = 3
        self.s.grid.cell_cnt_x = 4
        self.s.grid.cell_cnt_y = 5
        self.s.grid.cell_cnt_z = 6
        self.s.grid.n_gpus = tuple([1, 1, 1])
        self.s.grid.boundary_condition_x = grid.BoundaryCondition.PERIODIC
        self.s.grid.boundary_condition_y = grid.BoundaryCondition.PERIODIC
        self.s.grid.boundary_condition_z = grid.BoundaryCondition.PERIODIC
        self.s.solver = solver.YeeSolver()
        self.s.laser = None
        self.s.custom_user_input = None
        self.s.moving_window = None
        self.s.init_manager = species.InitManager()

        self.laser = GaussianLaser()
        self.laser.wavelength = 1.2
        self.laser.waist = 3.4
        self.laser.duration = 5.6
        self.laser.focus_pos = [0, 7.8, 0]
        self.laser.centroid_position = [0, 0, 0]
        self.laser.phase = 2.9
        self.laser.E0 = 9.0
        self.laser.pulse_init = 1.3
        self.laser.propagation_direction = [0, 1, 0]
        self.laser.polarization_type = GaussianLaser.PolarizationType.LINEAR
        self.laser.polarization_direction = [0, 0, 1]
        self.laser.laguerre_modes = [1.2, 2.4]
        self.laser.laguerre_phases = [2.4, 3.4]
        self.laser.huygens_surface_positions = [[1, -1], [1, -1], [1, -1]]

        self.customData_1 = [{"test_data_1": 1}, "tag_1"]
        self.customData_2 = [{"test_data_2": 2}, "tag_2"]

    def test_basic(self):
        s = self.s
        self.assertEqual(13.37, s.delta_t_si)
        self.assertEqual(42, s.time_steps)
        self.assertNotEqual(None, self.s.grid)

        # does not throw:
        s.get_rendering_context()

    def test_types(self):
        s = self.s
        with self.assertRaises(typeguard.TypeCheckError):
            s.delta_t_si = "1"
        with self.assertRaises(typeguard.TypeCheckError):
            s.time_steps = 14.3
        with self.assertRaises(typeguard.TypeCheckError):
            s.grid = [42, 13, 37]
        with self.assertRaises(typeguard.TypeCheckError):
            s.custom_user_input = {"test": 15}
        with self.assertRaises(typeguard.TypeCheckError):
            s.custom_user_input = customuserinput.CustomUserInput()

    def test_mandatory(self):
        # there are two main ways these objects are mandatory:
        # 1. they must be set at some point
        # 2. the can't be None (==can't be set to none)
        # option 1. is not tested, because this is ensured by the property
        # builder, # and the test code would be very boilerplate-y
        # option 2. is tested below:

        s = self.s
        with self.assertRaises(typeguard.TypeCheckError):
            s.delta_t_si = None
        with self.assertRaises(typeguard.TypeCheckError):
            s.time_steps = None
        with self.assertRaises(typeguard.TypeCheckError):
            s.grid = None

    def test_species_collision(self):
        """check that species name collisions are detected"""
        particle_1 = helper_get_species("collides")
        particle_2 = helper_get_species("collides")
        particle_3 = helper_get_species("doesnotcollide")

        expected_error_re = "^.*(collide|twice).*$"

        with self.assertRaisesRegex(Exception, expected_error_re):
            self.s.init_manager.all_species = [particle_1, particle_2]
            self.s.get_rendering_context()
        with self.assertRaisesRegex(Exception, expected_error_re):
            self.s.init_manager.all_species = [particle_1, particle_2, particle_3]
            self.s.get_rendering_context()
        with self.assertRaisesRegex(Exception, expected_error_re):
            self.s.init_manager.all_species = [particle_3, particle_3]
            self.s.get_rendering_context()

        # no errors:
        valid_species_lists = [[particle_2, particle_3], [particle_1, particle_3]]
        for valid_species_list in valid_species_lists:
            sim = deepcopy(self.s)
            # initmanager resets species attributes
            # -> no deepcopy necessary
            # (but still: species objects are SHARED between loop iterations)
            sim.init_manager.all_species = valid_species_list

            self.s.init_manager.all_operations = []
            for single_species in valid_species_list:
                not_placed = species.operation.NotPlaced()
                not_placed.species = single_species
                sim.init_manager.all_operations.append(not_placed)

                op_momentum = species.operation.SimpleMomentum()
                op_momentum.species = single_species
                op_momentum.drift = None
                op_momentum.temperature = None
                sim.init_manager.all_operations.append(op_momentum)

            # does not throw:
            sim.get_rendering_context()

    def test_get_rendering_context(self):
        """rendering context is returned"""
        # automatically checks & applies template
        self.assertTrue(isinstance(self.s, rendering.RenderedObject))

        # fill initmanager with some meaningful content
        species_dummy = species.Species()
        species_dummy.name = "myname"
        species_dummy.constants = []
        uniform_dist = species.operation.densityprofile.Uniform()
        uniform_dist.density_si = 123

        op_density = species.operation.SimpleDensity()
        op_density.ppc = 1
        op_density.profile = uniform_dist
        op_density.species = {
            species_dummy,
        }

        op_momentum = species.operation.SimpleMomentum()
        op_momentum.species = species_dummy
        op_momentum.drift = None
        op_momentum.temperature = None

        self.s.init_manager.all_species = [species_dummy]
        self.s.init_manager.all_operations = [op_density, op_momentum]

        context = self.s.get_rendering_context()

        # cross check with set values in setup method
        self.assertEqual(13.37, context["delta_t_si"])
        self.assertEqual(42, context["time_steps"])
        self.assertEqual("Yee", context["solver"]["name"])
        self.assertEqual(2, context["grid"]["cell_size"]["y"])
        self.assertEqual(None, context["laser"])
        self.assertEqual(self.s.init_manager.get_rendering_context(), context["species_initmanager"])
        self.assertEqual(1, context["output"]["auto"]["period"])

        self.assertNotEqual([], context["species_initmanager"]["species"])
        self.assertNotEqual([], context["species_initmanager"]["operations"])

    def test_laser_passthru(self):
        """laser is passed through"""
        # no laser
        self.assertEqual(None, self.s.laser)
        context = self.s.get_rendering_context()
        self.assertEqual(None, context["laser"])

        # a laser
        sim = self.s
        sim.laser = self.laser
        context = sim.get_rendering_context()
        laser_context = sim.laser.get_rendering_context()
        self.assertEqual(context["laser"], laser_context)

    def test_output_auto_short_duration(self):
        """period is always at least one"""
        for time_steps in [1, 17, 99]:
            self.s.time_steps = time_steps
            self.assertEqual(1, self.s.get_rendering_context()["output"]["auto"]["period"])

    def test_custom_input_pass_thru(self):
        i = customuserinput.CustomUserInput()

        i.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i.addToCustomInput(self.customData_2[0], self.customData_2[1])

        self.s.add_custom_user_input(i)

        renderingContextGoodResult = {"test_data_1": 1, "test_data_2": 2, "tags": ["tag_1", "tag_2"]}
        self.assertEqual(renderingContextGoodResult, self.s.get_rendering_context()["customuserinput"])

    def test_combination_of_several_custom_inputs(self):
        i_1 = customuserinput.CustomUserInput()
        i_2 = customuserinput.CustomUserInput()

        i_1.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i_2.addToCustomInput(self.customData_2[0], self.customData_2[1])

        self.s.add_custom_user_input(i_1)
        self.s.add_custom_user_input(i_2)

        renderingContextGoodResult = {"test_data_1": 1, "test_data_2": 2, "tags": ["tag_1", "tag_2"]}
        self.assertEqual(renderingContextGoodResult, self.s.get_rendering_context()["customuserinput"])

    def test_duplicated_tag_over_different_custom_inputs(self):
        i_1 = customuserinput.CustomUserInput()
        i_2 = customuserinput.CustomUserInput()

        i_1.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i_2.addToCustomInput(self.customData_2[0], self.customData_1[1])

        self.s.add_custom_user_input(i_1)
        self.s.add_custom_user_input(i_2)

        with self.assertRaisesRegex(ValueError, "duplicate tag provided!, tags must be unique!"):
            self.s.get_rendering_context()

    def test_duplicated_key_over_different_custom_inputs(self):
        i = customuserinput.CustomUserInput()
        i_sameValue = customuserinput.CustomUserInput()
        i_differentValue = customuserinput.CustomUserInput()

        duplicateKeyData_differentValue = {"test_data_1": 3}
        duplicateKeyData_sameValue = {"test_data_1": 1}

        i.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i_sameValue.addToCustomInput(duplicateKeyData_sameValue, "tag_2")
        i_differentValue.addToCustomInput(duplicateKeyData_differentValue, "tag_3")

        self.s.add_custom_user_input(i)

        # should work
        self.s.add_custom_user_input(i_sameValue)
        with self.assertRaisesRegex(ValueError, "Key test_data_1 exist already, and specified values differ."):
            self.s.add_custom_user_input(i_differentValue)
            self.s.get_rendering_context()
