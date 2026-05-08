"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from importlib.util import module_from_spec, spec_from_file_location
from operator import attrgetter
from pathlib import Path

import pytest
from picongpu import pypicongpu

EXAMPLES = list((Path(__file__).parents[3] / "examples").glob("*/main.py"))


@pytest.fixture(params=EXAMPLES, ids=list(map(attrgetter("parent.name"), EXAMPLES)))
def sim(request):
    """The simulation object defined in the corresponding example script."""
    module_spec = spec_from_file_location("example", request.param)
    module = module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.sim


@pytest.mark.skip("No interface exposed at the moment. Should be re-instantiated in some form.")
def test_compile_example(sim):
    """Attempts to compile the given simulation."""
    runner = pypicongpu.Runner(sim=sim)
    runner.generate(printDirToConsole=True)
    # The runner currently doesn't provide an interface for this.
    # runner.build()
