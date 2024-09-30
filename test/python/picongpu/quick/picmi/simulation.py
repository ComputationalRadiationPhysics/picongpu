"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from picongpu import picmi
from picongpu.pypicongpu import species, customuserinput
from picongpu.picmi.interaction.ionization.fieldionization import ADK, ADKVariant
from picongpu.picmi.interaction import Interaction

import unittest
import tempfile
import shutil
import os
import pathlib
import typeguard
import typing
import copy


@typeguard.typechecked
def get_grid(delta_x: float, delta_y: float, delta_z: float, n: int):
    # sets delta_[x,y,z] implicitly by providing bounding box+cell count
    return picmi.Cartesian3DGrid(
        number_of_cells=[n, n, n],
        lower_bound=[0, 0, 0],
        upper_bound=list(map(lambda x: n * x, [delta_x, delta_y, delta_z])),
        # required, otherwise won't spawn
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"],
    )


@typeguard.typechecked
def get_sim_cfl_helper(
    delta_t: typing.Optional[float],
    cfl: typing.Optional[float],
    delta_3d: typing.Tuple[float, float, float],
    method: str,
    n: int = 100,
) -> picmi.Simulation:
    grid = get_grid(delta_3d[0], delta_3d[1], delta_3d[2], n)
    solver = picmi.ElectromagneticSolver(method=method, grid=grid, cfl=cfl)
    return picmi.Simulation(time_step_size=delta_t, solver=solver)


class TestPicmiSimulation(unittest.TestCase):
    def __get_sim(self):
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver)

        return sim

    def __get_tmpdir_name(self):
        """
        get name of non-existing tmp dir which will be automatically cleaned up
        """
        name = None
        with tempfile.TemporaryDirectory() as tmpdir:
            name = tmpdir
        assert not os.path.exists(name)
        self.__to_cleanup.append(name)
        return name

    def setUp(self):
        self.sim = self.__get_sim()
        self.layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=2)
        self.__to_cleanup = []

        self.customData_1 = [{"test_data_1": 1}, "tag_1"]
        self.customData_2 = [{"test_data_2": 2}, "tag_2"]

    def tearDown(self):
        for dir_to_cleanup in self.__to_cleanup:
            if os.path.isdir(dir_to_cleanup):
                shutil.rmtree(dir_to_cleanup)
            assert not os.path.exists(dir_to_cleanup)

    def test_cfl_yee(self):
        # the Courant-Friedrichs-Lewy condition describes the relationship
        # between delta_t, delta_[x,y,z] and a parameter, here "cfl"
        # notably, all three can be given explicitly, though only two of the
        # three are required.
        # for practical reasons, delta_[x,y,z] has to be provided
        # this test checks the proper calculation of the cfl/delta_t

        # nothing defined if grid is empty
        sim = picmi.Simulation()
        self.assertEqual(None, sim.time_step_size)
        sim = picmi.Simulation(time_step_size=17)
        self.assertEqual(17, sim.time_step_size)

        # delta_t = cfl = None -> ignored (at least during instantiation;
        # can throw later)
        get_sim_cfl_helper(None, None, (1, 1, 1), "Yee")

        # delta_t -> cfl
        sim = get_sim_cfl_helper(2.02760320328617635877e-13, None, (7e-6, 8e-6, 9e-6), "Yee")
        self.assertAlmostEqual(13.37, sim.solver.cfl)

        # cfl -> delta_t
        sim = get_sim_cfl_helper(None, 0.99, (3, 4, 5), "Yee")
        self.assertAlmostEqual(7.14500557764070900528e-9, sim.time_step_size)

        # both delta_t and cfl defined:
        # case a: silently pass if they do match
        get_sim_cfl_helper(7.14500557764070900528e-9, 0.99, (3, 4, 5), "Yee")

        # case b: raise error if no match
        with self.assertRaises(ValueError):
            # delta_t does not match cfl at all
            get_sim_cfl_helper(1, 0.99, (3, 4, 5), "Yee")

    def test_cfl_not_yee(self):
        # if the solver is not yee, cfl and timestep can be set however
        # -> none of this raises an error
        get_sim_cfl_helper(7.14500557764070900528e-9, 0.99, (3, 4, 5), "CKC")
        get_sim_cfl_helper(42, 0.99, (3, 4, 5), "CKC")
        get_sim_cfl_helper(None, 0.99, (3, 4, 5), "CKC")
        get_sim_cfl_helper(42, None, (3, 4, 5), "CKC")
        get_sim_cfl_helper(None, None, (3, 4, 5), "CKC")

    def test_species_translation(self):
        """test that species are moved to PyPIConGPU simulation"""
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver)

        profile = picmi.UniformDistribution(density=42)
        layout3 = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        layout4 = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)

        # species list empty by default
        self.assertEqual([], sim.get_as_pypicongpu().init_manager.all_species)

        # not placed
        sim.add_species(picmi.Species(name="dummy1", mass=5), None)

        # placed with entire placement and 3ppc
        sim.add_species(picmi.Species(name="dummy2", mass=3, density_scale=4, initial_distribution=profile), layout3)

        # placed with default ratio of 1 and 4ppc
        sim.add_species(picmi.Species(name="dummy3", mass=3, initial_distribution=profile), layout4)

        picongpu = sim.get_as_pypicongpu()
        self.assertEqual(3, len(picongpu.init_manager.all_species))
        species_names = set(map(lambda species: species.name, picongpu.init_manager.all_species))
        self.assertEqual({"dummy1", "dummy2", "dummy3"}, species_names)

        # check typical ppc is derived
        self.assertEqual(picongpu.typical_ppc, 2)

    def test_explicit_typical_ppc(self):
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_typical_ppc=15)

        profile = picmi.UniformDistribution(density=42)
        layout3 = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        layout4 = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)

        # placed with entire placement and 3ppc
        sim.add_species(
            picmi.Species(name="dummy2", mass=3, charge=4, density_scale=4, initial_distribution=profile), layout3
        )
        # placed with default ratio of 1 and 4ppc
        sim.add_species(picmi.Species(name="dummy3", mass=3, charge=4, initial_distribution=profile), layout4)

        picongpu = sim.get_as_pypicongpu()
        self.assertEqual(2, len(picongpu.init_manager.all_species))
        species_names = set(map(lambda species: species.name, picongpu.init_manager.all_species))
        self.assertEqual({"dummy2", "dummy3"}, species_names)

        # check explicitly set typical ppc is respected
        self.assertEqual(picongpu.typical_ppc, 15)

    def test_wrong_explicitly_set_typical_ppc(self):
        grid = get_grid(1, 1, 1, 64)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)

        wrongValues = [0, -1, -15]
        for value in wrongValues:
            sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_typical_ppc=value)
            with self.assertRaisesRegex(ValueError, "typical_ppc must be >= 1"):
                sim.get_as_pypicongpu()

        wrongTypes = [0.0, -1.0, -15.0, 1.0, 15.0]
        for value in wrongTypes:
            with self.assertRaisesRegex(
                typeguard.TypeCheckError, '"picongpu_typical_ppc" .* did not match any element in the union'
            ):
                sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_typical_ppc=value)

    def test_invalid_placement(self):
        profile = picmi.UniformDistribution(density=42)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)

        # both profile and layout must be given
        with self.assertRaisesRegex(Exception, ".*initial.*distribution.*"):
            # no profile
            sim = copy.deepcopy(self.sim)
            sim.add_species(picmi.Species(name="dummy3"), layout)
            sim.get_as_pypicongpu()
        with self.assertRaisesRegex(Exception, ".*layout.*"):
            # no layout
            sim = copy.deepcopy(self.sim)
            sim.add_species(picmi.Species(name="dummy3", initial_distribution=profile), None)
            sim.get_as_pypicongpu()

        with self.assertRaisesRegex(Exception, ".*initial.*distribution.*"):
            # neither profile nor layout, but ratio
            sim = copy.deepcopy(self.sim)
            sim.add_species(picmi.Species(name="dummy3", density_scale=7), None)
            sim.get_as_pypicongpu()

    def test_operations_simple_density_translated(self):
        """simple density operations are correctly derived"""
        profile = picmi.UniformDistribution(density=42)
        other_profile = picmi.UniformDistribution(density=17)
        layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=3)
        other_layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=4)

        self.sim.add_species(
            picmi.Species(name="colocated1", mass=1, density_scale=4, initial_distribution=profile),
            layout,
        )
        self.sim.add_species(
            picmi.Species(name="colocated2", mass=2, density_scale=2, initial_distribution=profile),
            layout,
        )
        self.sim.add_species(
            picmi.Species(name="separate1", mass=3, initial_distribution=other_profile),
            layout,
        )
        self.sim.add_species(
            picmi.Species(name="separate2", mass=4, initial_distribution=profile),
            other_layout,
        )

        pypic = self.sim.get_as_pypicongpu()
        initmgr = pypic.init_manager

        # species
        self.assertEqual(4, len(initmgr.all_species))
        self.assertEqual(
            ["colocated1", "colocated2", "separate1", "separate2"],
            list(map(lambda species: species.name, initmgr.all_species)),
        )

        # operations
        density_operations = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleDensity),
                initmgr.all_operations,
            )
        )
        self.assertEqual(3, len(density_operations))
        for op in density_operations:
            self.assertTrue(isinstance(op.profile, species.operation.densityprofile.Uniform))

            # passes:
            op.check_preconditions()

            species_names = set(map(lambda species: species.name, op.species))

            # ensure grouping:
            if "separate1" in species_names or "separate2" in species_names:
                # one of the two lone species
                self.assertEqual(1, len(species_names))
            else:
                # the two colocated species
                self.assertEqual(2, len(species_names))

            # check profile
            if "separate2" in species_names or "colocated1" in species_names:
                # used "profile"
                self.assertEqual(42, op.profile.density_si)
            else:
                # used "other_profile"
                self.assertEqual(17, op.profile.density_si)

            # check layout
            if "separate1" in species_names or "colocated1" in species_names:
                # used "layout"
                self.assertEqual(3, op.ppc)
            else:
                # used "other_layout"
                self.assertEqual(4, op.ppc)

    def test_operation_not_placed_translated(self):
        """non-placed species are correctly translated"""
        self.sim.add_species(picmi.Species(name="notplaced", mass=1, initial_distribution=None), None)

        pypicongpu = self.sim.get_as_pypicongpu()

        self.assertEqual(1, len(pypicongpu.init_manager.all_species))
        # not placed, momentum (both initialize to empty)
        self.assertEqual(2, len(pypicongpu.init_manager.all_operations))

        notplaced_ops = list(
            filter(
                lambda op: isinstance(op, species.operation.NotPlaced),
                pypicongpu.init_manager.all_operations,
            )
        )

        self.assertEqual(1, len(notplaced_ops))
        self.assertEqual("notplaced", notplaced_ops[0].species.name)

    def test_operation_momentum(self):
        """operation for momentum correctly derived from species"""
        self.sim.add_species(
            picmi.Species(
                name="valid",
                mass=17,
                initial_distribution=picmi.UniformDistribution(
                    density=17,
                    rms_velocity=[17, 17, 17],
                    directed_velocity=[31283745.0, 45132121.0, 215484563.0],
                ),
            ),
            picmi.PseudoRandomLayout(n_macroparticles_per_cell=2),
        )

        pypicongpu = self.sim.get_as_pypicongpu()

        mom_ops = list(
            filter(
                lambda op: isinstance(op, species.operation.SimpleMomentum),
                pypicongpu.init_manager.all_operations,
            )
        )

        # momentum operation must always be generated
        self.assertEqual(1, len(mom_ops))
        mom_op = mom_ops[0]

        self.assertEqual("valid", mom_op.species.name)
        self.assertAlmostEqual(3.06645343e19, mom_op.temperature.temperature_kev, delta=1e13)
        self.assertEqual(
            mom_op.drift.direction_normalized,
            (0.14068221552237223, 0.2029580145696681, 0.9690286675623457),
        )
        self.assertAlmostEqual(1.491037242289643, mom_op.drift.gamma)

    def test_moving_window(self):
        """test that the user may set moving window"""
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            lower_bound=[0, 0, 0],
            upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-6],
            lower_boundary_conditions=["open", "open", "periodic"],
            upper_boundary_conditions=["open", "open", "periodic"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(
            time_step_size=1.39e-16, max_steps=int(2048), solver=solver, picongpu_moving_window_move_point=0.9
        )
        pypic = sim.get_as_pypicongpu()

        self.assertAlmostEqual(pypic.moving_window.move_point, 0.9)
        self.assertEqual(pypic.moving_window.stop_iteration, None)

    def test_add_ionization_model(self):
        """ionization model is added correctly"""
        e = picmi.Species(name="e", particle_type="electron")
        ion1 = picmi.Species(name="hydrogen", particle_type="H", charge_state=+1)
        ion2 = picmi.Species(name="nitrogen", particle_type="N", charge_state=+2)

        ionization_model_1 = ADK(
            ADK_variant=ADKVariant.LinearPolarization,
            ionization_current=None,
            ion_species=ion1,
            ionization_electron_species=e,
        )
        ionization_model_2 = ADK(
            ADK_variant=ADKVariant.LinearPolarization,
            ionization_current=None,
            ion_species=ion2,
            ionization_electron_species=e,
        )
        interaction = Interaction(ground_state_ionization_model_list=[ionization_model_1, ionization_model_2])

        sim = self.sim
        sim.add_species(e, None)
        sim.add_species(ion1, None)
        sim.add_species(ion2, None)

        # in use should be set via simulation constructor
        sim.picongpu_interaction = interaction

        pypic_sim = sim.get_as_pypicongpu()
        initmgr = pypic_sim.init_manager

        operation_types = list(map(lambda op: type(op), initmgr.all_operations))
        self.assertEqual(2, operation_types.count(species.operation.SetChargeState))

        for op in initmgr.all_operations:
            if isinstance(op, species.operation.SetChargeState) and op.species.name == "Nitrogen":
                self.assertEqual(5, op.bound_electrons)
            if isinstance(op, species.operation.SetChargeState) and op.species.name == "Hydrogen":
                self.assertEqual(0, op.bound_electrons)
            # other ops (position...): ignore

    def test_write_input_file(self):
        """sanity check picmi upstream: write input file"""
        sim = self.sim
        outdir = self.__get_tmpdir_name()
        self.assertTrue(not os.path.isdir(outdir))
        sim.write_input_file(outdir)
        self.assertTrue(os.path.isdir(outdir))
        self.assertTrue(os.path.exists(outdir + "/include/picongpu/param/simulation.param"))

    def test_custom_template_dir_basic_write_input_file(self):
        """providing custom template dir possible or write_input_file"""
        # note: automatically cleaned up in teardown
        out_dir = self.__get_tmpdir_name()

        with tempfile.TemporaryDirectory() as tmpdir:
            # create test template dir
            # -> use include/picongpu,
            #    because pic-create does not copy every dir
            os.makedirs(tmpdir + "/include/picongpu")
            with open(tmpdir + "/include/picongpu/time_steps.mustache", "w") as testfile:
                testfile.write("{{{time_steps}}}")

            grid = get_grid(1, 1, 1, 32)
            solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
            # explicitly set to None
            sim = picmi.Simulation(
                time_step_size=17,
                max_steps=128,
                solver=solver,
                picongpu_template_dir=tmpdir,
            )
            sim.write_input_file(out_dir)

        # check for generated (rendered) dir
        self.assertTrue(os.path.isfile(out_dir + "/include/picongpu/time_steps"))
        with open(out_dir + "/include/picongpu/time_steps") as rendered_file:
            self.assertEqual("128", rendered_file.read())

        # JSON has been dumped
        self.assertTrue(os.path.isfile(out_dir + "/pypicongpu.json"))

    def test_custom_input_basic_write_input_file(self):
        """test custom input may be rendered"""
        # note: automatically cleaned up in teardown
        out_dir = self.__get_tmpdir_name()

        # create bare bone PICMI-simulation
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        sim = picmi.Simulation(
            time_step_size=17,
            max_steps=128,
            solver=solver,
        )

        # add custom input
        i_1 = customuserinput.CustomUserInput()
        i_2 = customuserinput.CustomUserInput()

        i_1.addToCustomInput({"test_data_1": 1}, "tag_1")
        i_2.addToCustomInput({"test_data_2": 2}, "tag_2")

        sim.picongpu_add_custom_user_input(i_1)
        sim.picongpu_add_custom_user_input(i_2)

        # get pypicongpu simualtion
        pypicongpu_simulation = sim.get_as_pypicongpu()

        # write simulation
        sim.write_input_file(out_dir, pypicongpu_simulation=pypicongpu_simulation)

        # check for generated (rendered) dir
        self.assertTrue(os.path.isdir(out_dir))

        # JSON has been dumped
        self.assertTrue(os.path.isfile(out_dir + "/pypicongpu.json"))

    def test_custom_template_dir_basic_get_runner(self):
        """using picongpu_get_runner() directly sets template dir"""
        with tempfile.TemporaryDirectory() as tmpdir:
            grid = get_grid(1, 1, 1, 32)
            solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
            # explicitly set to None
            sim = picmi.Simulation(
                time_step_size=17,
                max_steps=128,
                solver=solver,
                picongpu_template_dir=tmpdir,
            )
            runner = sim.picongpu_get_runner()

            self.assertEqual(
                # note: this is the mangled name to access a private attribute
                # *actually* you should not access private variables,
                # however the alternative would be executing the runner,
                # which is very costly
                os.path.abspath(runner._Runner__pypicongpu_template_dir),
                os.path.abspath(tmpdir),
            )

    def test_custom_template_dir_optional(self):
        """custom template dir is optional"""
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        # explicitly set to None
        sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_template_dir=None)

        # simulation is valid
        self.assertNotEqual({}, self.sim.get_as_pypicongpu().get_rendering_context())
        runner = sim.picongpu_get_runner()

        # good default template dir is selected
        self.assertNotEqual(None, runner._Runner__pypicongpu_template_dir)
        self.assertNotEqual("", runner._Runner__pypicongpu_template_dir)

    def test_custom_template_dir_checks(self):
        """sanity checks are run on template dir"""
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)

        # empty string
        with self.assertRaisesRegex(Exception, ".*template.*"):
            picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, picongpu_template_dir="")

        # existing dir is ok:
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir_name = tmpdir
            sim = picmi.Simulation(
                time_step_size=17,
                max_steps=4,
                solver=solver,
                picongpu_template_dir=template_dir_name,
            )

            self.assertNotEqual({}, sim.get_as_pypicongpu().get_rendering_context())
            # no throw:
            sim.picongpu_get_runner()

        # left "with" block -- tmpdir is now deleted
        # -> now raises
        with self.assertRaisesRegex(Exception, ".*template.*"):
            picmi.Simulation(
                time_step_size=17,
                max_steps=4,
                solver=solver,
                picongpu_template_dir=template_dir_name,
            )

    def test_custom_template_dir_types(self):
        """custom template dir is typechecked"""
        grid = get_grid(1, 1, 1, 32)
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)

        valid_paths = [None, "/", pathlib.Path("/")]
        for valid_path in valid_paths:
            sim = picmi.Simulation(
                time_step_size=17,
                max_steps=4,
                solver=solver,
                picongpu_template_dir=valid_path,
            )
            self.assertNotEqual({}, sim.get_as_pypicongpu().get_rendering_context())
            # no throw:
            sim.picongpu_get_runner()

        invalid_paths = [1, ["/"], {}]
        for invalid_path in invalid_paths:
            with self.assertRaises(typeguard.TypeCheckError):
                picmi.Simulation(
                    time_step_size=17,
                    max_steps=4,
                    solver=solver,
                    picongpu_template_dir=invalid_path,
                )

    def test_custom_input_pass_thru(self):
        i = customuserinput.CustomUserInput()

        i.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i.addToCustomInput(self.customData_2[0], self.customData_2[1])

        self.sim.picongpu_add_custom_user_input(i)

        renderingContextGoodResult = {"test_data_1": 1, "test_data_2": 2, "tags": ["tag_1", "tag_2"]}
        self.assertEqual(
            renderingContextGoodResult, self.sim.get_as_pypicongpu().get_rendering_context()["customuserinput"]
        )

    def test_combination_of_several_custom_inputs(self):
        i_1 = customuserinput.CustomUserInput()
        i_2 = customuserinput.CustomUserInput()

        i_1.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i_2.addToCustomInput(self.customData_2[0], self.customData_2[1])

        self.sim.picongpu_add_custom_user_input(i_1)
        self.sim.picongpu_add_custom_user_input(i_2)

        renderingContextGoodResult = {"test_data_1": 1, "test_data_2": 2, "tags": ["tag_1", "tag_2"]}
        self.assertEqual(
            renderingContextGoodResult, self.sim.get_as_pypicongpu().get_rendering_context()["customuserinput"]
        )

    def test_duplicated_tag_over_different_custom_inputs(self):
        i_1 = customuserinput.CustomUserInput()
        i_2 = customuserinput.CustomUserInput()

        i_1.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i_2.addToCustomInput(self.customData_2[0], self.customData_1[1])

        self.sim.picongpu_add_custom_user_input(i_1)
        self.sim.picongpu_add_custom_user_input(i_2)

        with self.assertRaisesRegex(ValueError, "duplicate tag provided!, tags must be unique!"):
            self.sim.get_as_pypicongpu().get_rendering_context()

    def test_duplicated_key_over_different_custom_inputs(self):
        i = customuserinput.CustomUserInput()
        i_sameValue = customuserinput.CustomUserInput()
        i_differentValue = customuserinput.CustomUserInput()

        duplicateKeyData_differentValue = {"test_data_1": 3}
        duplicateKeyData_sameValue = {"test_data_1": 1}

        i.addToCustomInput(self.customData_1[0], self.customData_1[1])
        i_sameValue.addToCustomInput(duplicateKeyData_sameValue, "tag_2")
        i_differentValue.addToCustomInput(duplicateKeyData_differentValue, "tag_3")

        self.sim.picongpu_add_custom_user_input(i)

        # should work
        self.sim.picongpu_add_custom_user_input(i_sameValue)
        self.sim.get_as_pypicongpu().get_rendering_context()

        with self.assertRaisesRegex(ValueError, "Key test_data_1 exist already, and specified values differ."):
            self.sim.picongpu_add_custom_user_input(i_differentValue)
            self.sim.get_as_pypicongpu().get_rendering_context()
