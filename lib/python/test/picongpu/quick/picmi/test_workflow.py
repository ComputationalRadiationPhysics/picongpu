"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import json

from cwltool.context import RuntimeContext
from cwltool.factory import Factory, WorkflowStatus
from picongpu.picmi import Cartesian3DGrid, ElectromagneticSolver, Simulation
from pytest import fixture, raises


@fixture
def sim():
    number_of_cells = 32
    cell_size = 1
    sim = Simulation(
        time_step_size=17,
        max_steps=4,
        solver=ElectromagneticSolver(
            method="Yee",
            grid=Cartesian3DGrid(
                number_of_cells=[number_of_cells, number_of_cells, number_of_cells],
                lower_bound=[0, 0, 0],
                upper_bound=list(map(lambda x: number_of_cells * x, [cell_size, cell_size, cell_size])),
                # required, otherwise won't spawn
                lower_boundary_conditions=["open", "open", "periodic"],
                upper_boundary_conditions=["open", "open", "periodic"],
            ),
        ),
    )
    sim.picongpu_get_runner().generate()
    return sim


@fixture
def workflow_definition_path(sim):
    return sim.picongpu_get_runner().workflow_definition_path


@fixture
def workflow_input(sim):
    with sim.picongpu_get_runner().workflow_input_path.open("r") as file:
        return json.load(file)


def test_validate_workflow(workflow_definition_path, workflow_input):
    # Couldn't have come up with a stranger interface:
    # The `validate_only` mode of the factory uses an exception to shortcircuit apparently.
    # Well, in this case "success" means:
    with raises(WorkflowStatus, match="Completed ValidationSuccess"):
        Factory(runtime_context=RuntimeContext(kwargs={"validate_only": True})).make(str(workflow_definition_path))(
            **workflow_input
        )
