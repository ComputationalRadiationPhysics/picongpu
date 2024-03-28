"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.runner import Runner

import unittest
import typeguard

from picongpu import picmi

import tempfile
import os
import re
import pathlib
import json


class TestRunner(unittest.TestCase):
    # note: tests involving actual running/building are in the long tests

    def __get_tmpdir_name(self):
        name = None
        with tempfile.TemporaryDirectory() as tmpdir:
            name = tmpdir
        return name

    def setUp(self):
        # Note that this picmi simulation is a valid specification,
        # but -- at the time of writing -- does not successfully build.
        # the long tests (which actually build stuff, this here doesn't)
        # have a working parameter set
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            lower_bound=[0, 0, 0],
            upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-6],
            lower_boundary_conditions=["open", "open", "periodic"],
            upper_boundary_conditions=["open", "open", "periodic"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
        self.picmi_sim = picmi.Simulation(time_step_size=1.39e-16, max_steps=int(2048), solver=solver)
        self.sim = self.picmi_sim.get_as_pypicongpu()

        # reset default scratch dir
        if Runner.SCRATCH_ENV_NAME in os.environ:
            del os.environ[Runner.SCRATCH_ENV_NAME]

        self.tmpdir = tempfile.gettempdir()
        assert os.path.isdir(self.tmpdir)

        # use as working dir
        os.chdir(self.tmpdir)

        self.existing_dir1 = self.__get_tmpdir_name()
        self.existing_dir2 = self.__get_tmpdir_name()
        os.mkdir(self.existing_dir1)
        os.mkdir(self.existing_dir2)
        self.nonexisting_dir1 = self.__get_tmpdir_name()
        self.nonexisting_dir2 = self.__get_tmpdir_name()
        self.nonexisting_dir3 = self.__get_tmpdir_name()

        self.nonexisting_relative_dir1 = "BHJADSdir1"
        self.nonexisting_relative_dir2 = "BHJADSdir2"
        self.existing_relative_dir1 = "NBJCMACBAkjsdir1"
        os.mkdir(self.existing_relative_dir1)

        self.__maybe_tmpdir_destroy = [
            self.existing_dir1,
            self.existing_dir2,
            os.path.abspath(self.existing_relative_dir1),
        ]

        assert Runner.SCRATCH_ENV_NAME not in os.environ
        assert os.path.isdir(self.tmpdir)
        assert os.path.isdir(self.existing_dir1)
        assert os.path.isdir(self.existing_dir2)
        assert not os.path.exists(self.nonexisting_dir1)
        assert not os.path.exists(self.nonexisting_dir2)
        assert not os.path.exists(self.nonexisting_dir3)
        assert not os.path.exists(self.nonexisting_relative_dir1)
        assert not os.path.exists(self.nonexisting_relative_dir2)
        assert os.path.isdir(self.existing_relative_dir1)

    def tearDown(self):
        for d in self.__maybe_tmpdir_destroy:
            if os.path.isdir(d):
                os.rmdir(d)
            assert not os.path.exists(d)

        assert os.path.isdir(self.tmpdir)
        assert not os.path.exists(self.existing_dir1)
        assert not os.path.exists(self.existing_dir2)
        assert not os.path.exists(self.nonexisting_dir1)
        assert not os.path.exists(self.nonexisting_dir2)
        assert not os.path.exists(self.nonexisting_dir3)

    def test_test_setup(self):
        """check that test setup works"""
        # does not throw
        Runner(self.picmi_sim.get_as_pypicongpu())
        Runner(self.sim)

        r = Runner(
            self.sim,
            scratch_dir=self.existing_dir1,
            setup_dir=self.nonexisting_dir1,
            run_dir=self.nonexisting_dir2,
        )
        self.assertEqual(self.existing_dir1, r.scratch_dir)
        self.assertEqual(self.nonexisting_dir1, r.setup_dir)
        self.assertEqual(self.nonexisting_dir2, r.run_dir)

    def test_param_types(self):
        """
        check that arbitrary types are not supported
        (also sse test_init_sim_type below)
        """
        # also check that correct usage works:
        Runner(self.sim)

        with self.assertRaises(typeguard.TypeCheckError):
            Runner(self.sim, pypicongpu_template_dir=1)
        with self.assertRaises(typeguard.TypeCheckError):
            Runner(self.sim, scratch_dir=["/", "tmp"])
        with self.assertRaises(typeguard.TypeCheckError):
            Runner(self.sim, setup_dir={})
        with self.assertRaises(typeguard.TypeCheckError):
            Runner(self.sim, run_dir=lambda x: x)

        r = Runner(self.sim)
        with self.assertRaises(typeguard.TypeCheckError):
            r.setup_dir = 1
        with self.assertRaises(typeguard.TypeCheckError):
            r.run_dir = []
        with self.assertRaises(typeguard.TypeCheckError):
            r.scratch_dir = {}

    def test_dir_collision(self):
        """check that dir names are different"""
        # others empty: does not throw
        Runner(self.sim, run_dir="abc")
        Runner(self.sim, setup_dir="abc")
        Runner(self.sim, scratch_dir=self.existing_dir1)

        # actual collision tests
        # note: do not check the message content here
        # -> the error *might* be collision,
        # but also might be due to (non-) existing contraints for dirs
        with self.assertRaises(Exception):
            Runner(
                self.sim,
                run_dir=self.nonexisting_dir1,
                scratch_dir=self.nonexisting_dir1,
            )
        with self.assertRaises(Exception):
            Runner(self.sim, run_dir=self.nonexisting_dir1, setup_dir=self.nonexisting_dir1)
        with self.assertRaises(Exception):
            Runner(
                self.sim,
                scratch_dir=self.nonexisting_dir1,
                setup_dir=self.nonexisting_dir1,
            )
        with self.assertRaises(Exception):
            Runner(
                self.sim,
                scratch_dir=self.nonexisting_dir1,
                setup_dir=self.nonexisting_dir1,
                run_dir=self.nonexisting_dir1,
            )

    def test_scratch_from_env(self):
        """test that the scratch dir is loaded from environment when None"""
        r = Runner(self.sim)
        self.assertEqual(None, r.scratch_dir)

        r = Runner(self.sim, scratch_dir=self.existing_dir1)
        self.assertEqual(self.existing_dir1, r.scratch_dir)
        self.assertTrue(r.run_dir.startswith(r.scratch_dir))

        # now provide default via environment
        os.environ[Runner.SCRATCH_ENV_NAME] = self.existing_dir2

        r = Runner(self.sim)
        self.assertEqual(self.existing_dir2, r.scratch_dir)
        self.assertTrue(r.run_dir.startswith(r.scratch_dir))

        # can still be overwritten though
        r = Runner(self.sim, scratch_dir=self.existing_dir1)
        self.assertEqual(self.existing_dir1, r.scratch_dir)
        self.assertTrue(r.run_dir.startswith(r.scratch_dir))

        # relative path is accepted, but converted to absolute
        os.environ[Runner.SCRATCH_ENV_NAME] = self.existing_relative_dir1
        r = Runner(self.sim)
        self.assertEqual(
            os.path.realpath(self.existing_relative_dir1),
            os.path.realpath(r.scratch_dir),
        )
        self.assertTrue(os.path.isabs(r.scratch_dir))

    def test_missing_dirs_generated(self):
        """check that given dirs are kept and not-given dirs generated"""

        def check_postconditions(r):
            self.assertTrue(not os.path.exists(r.setup_dir))
            self.assertTrue(not os.path.exists(r.run_dir))

        # all dirs missing
        r = Runner(self.sim)
        check_postconditions(r)
        # all generated inside tmp dir
        self.assertTrue(r.setup_dir.startswith(self.tmpdir))
        self.assertTrue(r.run_dir.startswith(self.tmpdir))

        # run derived from scratch
        r = Runner(self.sim, scratch_dir=self.existing_dir1)
        check_postconditions(r)
        self.assertTrue(r.run_dir.startswith(r.scratch_dir))
        self.assertTrue(r.setup_dir.startswith(self.tmpdir))

        # scratch ignore if run is given
        r = Runner(self.sim, scratch_dir=self.existing_dir1, run_dir=self.nonexisting_dir1)
        check_postconditions(r)
        self.assertTrue(not r.setup_dir.startswith(r.scratch_dir))
        self.assertEqual(self.nonexisting_dir1, r.run_dir)

        # setup kept as given
        r = Runner(self.sim, setup_dir=self.nonexisting_dir2)
        check_postconditions(r)
        self.assertEqual(self.nonexisting_dir2, r.setup_dir)

    def test_checks_given_dirs(self):
        """sanity checks performed on given dirs"""
        # valid call
        Runner(
            self.sim,
            scratch_dir=self.existing_dir1,
            setup_dir=self.nonexisting_dir1,
            run_dir=self.nonexisting_dir2,
        )

        # make sure that the paths are not used/created during __init__
        assert os.path.isdir(self.existing_dir1)
        assert os.path.isdir(self.existing_dir2)
        assert not os.path.exists(self.nonexisting_dir1)
        assert not os.path.exists(self.nonexisting_dir2)
        assert not os.path.exists(self.nonexisting_dir3)

        # invalid calls:
        with self.assertRaisesRegex(Exception, ".*scratch.*"):
            Runner(
                self.sim,
                scratch_dir=self.nonexisting_dir3,
                setup_dir=self.nonexisting_dir1,
                run_dir=self.nonexisting_dir2,
            )
        with self.assertRaisesRegex(Exception, ".*setup.*"):
            Runner(
                self.sim,
                scratch_dir=self.existing_dir1,
                setup_dir=self.existing_dir2,
                run_dir=self.nonexisting_dir2,
            )
        with self.assertRaisesRegex(Exception, ".*run.*"):
            Runner(
                self.sim,
                scratch_dir=self.existing_dir1,
                setup_dir=self.nonexisting_dir1,
                run_dir=self.existing_dir2,
            )

    def test_invalid_dir_names(self):
        """forbidden character (not alphanum.-_) produce an error"""
        allowed_dir_names = [
            "abas",
            "-123",
            "123",
            "/ahjsd",
            "hnaxbcnxyci8HJBASDJASG61723",
            "/tmp/hadjs/7123",
        ]
        forbidden_dir_names = [
            "",
            ";asd",
            "&/(12)",
            "try#123",
            "a:colon",
            "why is space not allowed",
        ]

        for allowed_name in allowed_dir_names:
            # no error
            Runner(self.sim, setup_dir=allowed_name)
            Runner(self.sim, run_dir=allowed_name)
            # do not check with scratch dir, as it must not necessarily exist

        for forbidden_name in forbidden_dir_names:
            with self.assertRaisesRegex(Exception, ".*valid.*"):
                Runner(self.sim, scratch_dir=forbidden_name)
            with self.assertRaisesRegex(Exception, ".*valid.*"):
                Runner(self.sim, setup_dir=forbidden_name)
            with self.assertRaisesRegex(Exception, ".*valid.*"):
                Runner(self.sim, run_dir=forbidden_name)

    def test_absolute(self):
        r = Runner(
            self.sim,
            scratch_dir=self.existing_relative_dir1,
            setup_dir=self.nonexisting_relative_dir1,
            run_dir=self.nonexisting_relative_dir2,
        )

        self.assertNotEqual(self.existing_relative_dir1, r.scratch_dir)
        self.assertNotEqual(self.nonexisting_relative_dir1, r.setup_dir)
        self.assertNotEqual(self.nonexisting_relative_dir2, r.run_dir)

        # note: realpath for existing dir
        self.assertEqual(
            os.path.realpath(self.existing_relative_dir1),
            os.path.realpath(r.scratch_dir),
        )
        self.assertEqual(
            os.path.abspath(self.nonexisting_relative_dir1),
            os.path.abspath(r.setup_dir),
        )
        self.assertEqual(os.path.abspath(self.nonexisting_relative_dir2), os.path.abspath(r.run_dir))

        self.assertTrue(os.path.isabs(r.scratch_dir))
        self.assertTrue(os.path.isabs(r.setup_dir))
        self.assertTrue(os.path.isabs(r.run_dir))

    def test_human_readable(self):
        """check that autogenerated names are (somewhat) human-readable"""
        r = Runner(self.sim)

        setup_dir_base = os.path.basename(r.setup_dir)
        run_dir_base = os.path.basename(r.run_dir)

        self.assertTrue(re.match("^pypicongpu-.*$", setup_dir_base))
        self.assertTrue("setup" in setup_dir_base)

        self.assertTrue(re.match("^pypicongpu-.*$", run_dir_base))
        self.assertTrue("run" in run_dir_base)

        # Check that both have a common prefix (besides pypicongpu-),
        # typically the date.
        # This is useful so dirs generated by the same runner
        # are next to each other when sorting.
        def get_wo_pypicongpu_prefix(s):
            m = re.match("^pypicongpu-(.*)$", s)
            # require a match
            assert m
            return m[1]

        self.assertEqual("123-teststring", get_wo_pypicongpu_prefix("pypicongpu-123-teststring"))

        setup_dir_base_noprefix = get_wo_pypicongpu_prefix(setup_dir_base)
        run_dir_base_noprefix = get_wo_pypicongpu_prefix(run_dir_base)

        self.assertEqual("blasabbl", os.path.commonprefix(["blasabbl123", "blasabblajhsdkljh"]))
        # common_start would typically be the current date
        # (though using the date is not required)
        common_start = os.path.commonprefix([setup_dir_base_noprefix, run_dir_base_noprefix])
        # six: shortest useful date representation YYMMDD
        self.assertTrue(len(common_start) >= 6)

    def test_init_picmi_or_picongpu(self):
        """
        check that the simulation can be initialized using
        both picmi or a pypicongpu object
        """
        # note: the other tests typically use the picongpu simulation,
        # so here emphasis is put on the picmi simulation object

        def is_sims_equal(a, b):
            """compare two pypicongpu simulations"""
            return a.get_rendering_context() == b.get_rendering_context()

        # check precondition by setup
        assert is_sims_equal(self.picmi_sim.get_as_pypicongpu(), self.sim)

        r_from_picmi = Runner(self.picmi_sim)
        r_from_picongpu = Runner(self.sim)

        self.assertTrue(is_sims_equal(r_from_picmi.sim, r_from_picongpu.sim))
        # sanity checks if everything worked
        self.assertEqual(r_from_picmi.scratch_dir, r_from_picongpu.scratch_dir)
        self.assertNotEqual(None, r_from_picmi.setup_dir)
        self.assertNotEqual(None, r_from_picmi.run_dir)

    def test_init_sim_type(self):
        """
        due to (potentially) circular imports,
        the runner __init__ can't use typeguard to check the simulation type.
        Instead manual type checks are used.
        This tests if these typechecks are implemented correctly.
        """
        # must work
        Runner(self.sim)
        Runner(self.picmi_sim)

        for invalid_sim in [None, {}, 0, ""]:
            with self.assertRaises(typeguard.TypeCheckError):
                Runner(invalid_sim)

    def test_applies_templates(self):
        """runner renders templates from template dir"""
        # generate test template
        with tempfile.TemporaryDirectory() as template_dir:
            template_path = pathlib.Path(template_dir)
            # note: put in subdir, b/c not all directories are cloned by
            # pic-create
            os.makedirs(template_path / "etc" / "picongpu")
            testfile_template = template_path / "etc" / "picongpu" / "date.mustache"
            with open(testfile_template, "w") as tpl_file:
                tpl_file.write("{{{_date}}}")
            # workaround (TODO rm): add location for pypicongpu.param
            os.makedirs(template_path / "include" / "picongpu" / "param")

            # create ruunner with previous tempalte dir, rest of directories
            # is not predefined
            runner = Runner(self.sim, pypicongpu_template_dir=template_dir)

            runner.generate()

            setup_path = pathlib.Path(runner.setup_dir)

            testfile_rendered = setup_path / "etc" / "picongpu" / "date"
            with open(testfile_rendered, "r") as rendered_file:
                content = rendered_file.read()
                # render  and template did something
                self.assertTrue("_date" not in content)
                self.assertNotEqual("{{{_date}}}", content)

                # did not replace with empty (i.e. _date was defined)
                self.assertNotEqual("", content)

    def test_dumps_rendering_context(self):
        """rendering context is dumped on generation"""
        runner = Runner(self.sim)
        runner.generate()
        setup_path = pathlib.Path(runner.setup_dir)
        dump_file = setup_path / "pypicongpu.json"

        self.assertTrue(dump_file.exists())
        with open(dump_file, "r") as file:
            content = file.read()
            self.assertNotEqual("", content)
            from_file = json.loads(content)
            self.assertEqual(self.sim.get_rendering_context(), from_file)
