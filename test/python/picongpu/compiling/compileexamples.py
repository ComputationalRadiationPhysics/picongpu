"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from picongpu import pypicongpu

import importlib.util
import os

import unittest


class TestExamples(unittest.TestCase):
    def load_example_script(self, path):
        """load and execute example PICMI script from given path"""
        module_spec = importlib.util.spec_from_file_location("example", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)

        sim = module.sim

        return sim

    def build_simulation(self, sim):
        """build the given instance of simulation"""
        runner = pypicongpu.Runner(sim)
        runner.generate(printDirToConsole=True)
        runner.build()

    def test_LWFA_example(self):
        """generate a PIConGPU setup from the laser_wakefield PICMI example and compile the setup"""
        sim = self.load_example_script(
            os.environ["PICSRC"] + "/share/picongpu/pypicongpu/examples/laser_wakefield/main.py"
        )

        self.build_simulation(sim)

    def test_warm_plasma_example(self):
        """generate a PIConGPU setup from the warm_plasma PICMI example and compile the setup"""
        sim = self.load_example_script(os.environ["PICSRC"] + "/share/picongpu/pypicongpu/examples/warm_plasma/main.py")

        self.build_simulation(sim)
