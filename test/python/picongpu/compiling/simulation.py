"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu import pypicongpu, picmi

from typeguard import typechecked

import unittest
import tempfile
import os


@typechecked
def get_grid(delta_x: float, delta_y: float, delta_z: float, n: int):
    # sets delta_[x,y,z] implicitly by providing bounding box+cell count
    return picmi.Cartesian3DGrid(
        number_of_cells=[n, n, n],
        lower_bound=[0, 0, 0],
        upper_bound=list(map(lambda x: n*x,
                             [delta_x, delta_y, delta_z])),
        # required, otherwise won't spawn
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"])


class TestSimulation(unittest.TestCase):
    def test_minimal(self):
        """smallest possible example"""
        sim = pypicongpu.Simulation()
        sim.delta_t_si = 1.39e-16
        sim.time_steps = 1
        sim.grid = pypicongpu.grid.Grid3D()
        sim.grid.cell_size_x_si = 1.776e-07
        sim.grid.cell_size_y_si = 4.43e-08
        sim.grid.cell_size_z_si = 1.776e-07
        sim.grid.cell_cnt_x = 1
        sim.grid.cell_cnt_y = 1
        sim.grid.cell_cnt_z = 1
        sim.grid.n_gpus = (1, 1, 1)
        sim.grid.boundary_condition_x = \
            pypicongpu.grid.BoundaryCondition.PERIODIC
        sim.grid.boundary_condition_y = \
            pypicongpu.grid.BoundaryCondition.PERIODIC
        sim.grid.boundary_condition_z = \
            pypicongpu.grid.BoundaryCondition.PERIODIC
        sim.laser = None
        sim.solver = pypicongpu.solver.YeeSolver()
        sim.init_manager = pypicongpu.species.InitManager()

        runner = pypicongpu.Runner(sim)
        runner.generate(printDirToConsole=True)
        runner.build()
        runner.run()

    def test_custom_template_dir(self):
        """may pass custom template dir"""

        # note: is not required to compile
        # this test checks if the correct dir is **passed to the runner**
        # (instead of checking if the correct files have been generated)

        with tempfile.TemporaryDirectory() as tmpdir:
            grid = get_grid(1, 1, 1, 32)
            solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
            # explicitly set to None
            sim = picmi.Simulation(time_step_size=17,
                                   max_steps=128,
                                   solver=solver,
                                   picongpu_template_dir=tmpdir)

            # there is no code -> this should not compile
            with self.assertRaises(Exception):
                sim.picongpu_run()

            template_dir_name = tmpdir

        runner = sim.picongpu_get_runner()

        # check for generated (rendered) dir
        self.assertTrue(os.path.isdir(runner.setup_dir))
        self.assertEqual(
            os.path.abspath(template_dir_name),
            os.path.abspath(runner._Runner__pypicongpu_template_dir))
