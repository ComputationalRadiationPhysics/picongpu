"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.runner import Runner

import unittest

from picongpu import picmi

import tempfile
import os
import shutil


class TestRunner(unittest.TestCase):
    # note: these tests run a long time
    # (as they involve actually trying to build/run stuff)
    # for short tests see the short/ directory

    # note: this took approx. 20 min to complete on a dedicated build machine

    def __get_tmpdir_name(self):
        name = None
        with tempfile.TemporaryDirectory() as tmpdir:
            name = tmpdir
        assert not os.path.exists(name)
        return name

    def setUp(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[192, 2048, 12],
            lower_bound=[0, 0, 0],
            upper_bound=[3.40992e-5, 9.07264e-5, 2.1312e-6],
            lower_boundary_conditions=["open", "open", "periodic"],
            upper_boundary_conditions=["open", "open", "periodic"])
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid, cfl=0.99)
        self.picmi_sim = picmi.Simulation(max_steps=int(1), solver=solver)
        self.sim = self.picmi_sim.get_as_pypicongpu()

        # unset default scratch dir if set
        if Runner.SCRATCH_ENV_NAME in os.environ:
            del os.environ[Runner.SCRATCH_ENV_NAME]

        self.nonexisting_dir = self.__get_tmpdir_name()
        self.nonexisting_dir2 = self.__get_tmpdir_name()

    def tearDown(self):
        for dir in [self.nonexisting_dir, self.nonexisting_dir2]:
            if os.path.isdir(dir):
                shutil.rmtree(dir)
            assert not os.path.exists(dir)

    def test_step_order_required(self):
        """check that calling steps out of order does not work"""
        r = Runner(self.sim)

        # general pattern:
        # The steps generate(), build(), run() are allowed
        # exactly once, and in that order.
        # Any other call (or calling the steps again) is prohibited
        # and should produce an error.

        with self.assertRaises(Exception):
            r.build()
        with self.assertRaises(Exception):
            r.run()

        # no raise
        r.generate()
        with self.assertRaises(Exception):
            r.generate()
        with self.assertRaises(Exception):
            r.run()

        r.build()
        with self.assertRaises(Exception):
            r.generate()
        with self.assertRaises(Exception):
            r.build()

        r.run()
        with self.assertRaises(Exception):
            r.generate()
        with self.assertRaises(Exception):
            r.build()
        with self.assertRaises(Exception):
            r.run()

    def test_generate_code(self):
        # no error
        r = Runner(self.sim, setup_dir=self.nonexisting_dir)
        # now the dir is created in the meantime -> causes error
        assert not os.path.exists(r.setup_dir)
        os.mkdir(r.setup_dir)
        assert os.path.isdir(r.setup_dir)

        # error because now path exists
        with self.assertRaises(Exception):
            r.generate()

    def test_build(self):
        # no error
        r = Runner(self.sim, setup_dir=self.nonexisting_dir)
        r.generate()

        # now the dir is created in the meantime -> causes error
        pic_build_dir = os.path.join(r.setup_dir, ".build")
        assert not os.path.exists(pic_build_dir)
        os.mkdir(pic_build_dir)
        assert os.path.isdir(pic_build_dir)

        # error because now path+.build (pic-build destination) exists
        with self.assertRaises(Exception):
            r.build()

        # but: injecting a generated (but not built) dir
        # from another runner works
        other_runner = Runner(self.sim)
        other_runner.generate(printDirToConsole=True)

        r.setup_dir = other_runner.setup_dir
        assert os.path.isdir(r.setup_dir)

        # no error
        r.build()

    def test_run(self):
        # no error
        r = Runner(self.sim, run_dir=self.nonexisting_dir)
        r.generate()
        r.build()

        # now the dir is created in the meantime -> causes error
        assert not os.path.exists(r.run_dir)
        os.mkdir(r.run_dir)
        assert os.path.isdir(r.run_dir)

        # error because now path exists
        with self.assertRaises(Exception):
            r.run()

        # but: resetting the path to sth nonexistent works
        r.run_dir = self.nonexisting_dir2
        assert not os.path.exists(r.run_dir)
        r.run()

    def test_dir_reset_generate(self):
        """
        test what happens if dir vars are reset after correct construction
        """
        r = Runner(self.sim)
        # other tests ensure that vars are set

        with self.assertRaises(Exception):
            r.setup_dir = None
            r.generate()

    def test_dir_reset_build(self):
        r = Runner(self.sim)
        r.generate()
        with self.assertRaises(Exception):
            r.setup_dir = None
            r.build()

    def test_dir_reset_run(self):
        with tempfile.TemporaryDirectory() as existing_dir:
            r = Runner(self.sim, scratch_dir=existing_dir)
            r.generate()
            r.build()

            # scratch dir is set
            # -> run dir could *theoretically* be guessed again
            # (but this is not the case)
            assert os.path.isdir(r.scratch_dir)
            self.assertTrue(r.run_dir.startswith(r.scratch_dir))

            with self.assertRaises(Exception):
                r.run_dir = None
                r.run()

    def test_dir_rm_scratch(self):
        """
        resetting the scratch dir after run dir has been set has no effect
        """
        with tempfile.TemporaryDirectory() as existing_dir:
            r = Runner(self.sim, scratch_dir=existing_dir)
            r.generate()
            r.build()

            r.scratch_dir = None
            self.assertTrue(r.run_dir is not None)
            # no error
            r.run()

    def test_dirs_used(self):
        r = Runner(self.sim)

        self.assertTrue(not os.path.exists(r.run_dir))
        self.assertTrue(not os.path.exists(r.setup_dir))
        r.generate()
        self.assertTrue(not os.path.exists(r.run_dir))
        self.assertTrue(os.path.isdir(r.setup_dir))

        r.build()
        self.assertTrue(not os.path.exists(r.run_dir))
        self.assertTrue(os.path.isdir(r.setup_dir))

        r.run()
        self.assertTrue(os.path.isdir(r.run_dir))
        self.assertTrue(os.path.isdir(r.setup_dir))
