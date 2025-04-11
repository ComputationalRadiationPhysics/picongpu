"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from picongpu import pypicongpu

import importlib.util
import os
from pathlib import Path

import unittest

EXAMPLES = filter(
    Path.is_dir,
    (Path(os.environ["PICSRC"]) / "share/picongpu/pypicongpu/examples/").iterdir(),
)


class TestExamplesMeta(type):
    def __new__(cls, name, bases, dict):
        # Generate one test for each example in the examples folder
        for example in EXAMPLES:
            name = "test_" + example.name
            dict[name] = (
                # This is slightly convoluted:
                # Python's semantics around variables implement
                # "sharing" semantics (not even quite reference semantics).
                # Also, lambdas capture the variable and not the value.
                # So after the execution of a loop all lambdas refer to
                # the last value of the loop variable
                # if they tried to capture it.
                # So, we need to eagerly evaluate the `example` variable
                # which we achieve via an immediately evaluated lambda expression.
                # Please excuse my C++ dialect.
                lambda example: lambda self: self.build_simulation(self.load_example_script(example / "main.py"))
            )(example)
        return type.__new__(cls, name, bases, dict)


class TestExamples(unittest.TestCase, metaclass=TestExamplesMeta):
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
