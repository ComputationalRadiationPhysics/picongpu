"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
from functools import reduce
from pathlib import Path
from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    GaussianLaser,
    Simulation,
    constants,
)
from picongpu.picmi import Species as Species
from picongpu.picmi.diagnostics import Checkpoint, TimeStepSpec
from picongpu.picmi.lasers import PolarizationType

from .compare_particles import read_fields

logging.basicConfig(level=logging.INFO)

STEPS = 300
LOWER_BOUNDARY = np.zeros(3)
NUMBER_OF_CELLS = np.array([192, 128, 192])
CELL_SIZE = np.array([0.1772e-6, 0.4430e-7, 0.1772e-6])  # unit: meter
UPPER_BOUNDARY = NUMBER_OF_CELLS * CELL_SIZE + LOWER_BOUNDARY

GRID = Cartesian3DGrid(
    number_of_cells=NUMBER_OF_CELLS,
    lower_bound=LOWER_BOUNDARY,
    upper_bound=UPPER_BOUNDARY,
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
SOLVER = ElectromagneticSolver(grid=GRID, method="Yee", cfl=0.9)


PULSE_INIT = 15.0
LASER_DURATION = 5.0e-15
FOCAL_POSITION = NUMBER_OF_CELLS / 2 * CELL_SIZE
FOCAL_POSITION[1] = 4.62e-5
CENTROID_POSITION = NUMBER_OF_CELLS / 2 * CELL_SIZE
CENTROID_POSITION[1] = -0.5 * PULSE_INIT * LASER_DURATION * constants.c

LASERS = [
    GaussianLaser(
        wavelength=0.8e-6,
        waist=5.0e-6 / 1.17741,
        duration=LASER_DURATION,
        propagation_direction=[0.0, 1.0, 0.0],
        polarization_direction=[1.0, 0.0, 0.0],
        focal_position=FOCAL_POSITION,
        centroid_position=CENTROID_POSITION,
        picongpu_polarization_type=PolarizationType.LINEAR,
        a0=8.0,
        phi0=0.0,
    )
]


def basic_simulation():
    return Simulation(max_steps=STEPS, solver=SOLVER)


RUN_DIR = "/tmp/pypicongpu-2026-02-25-11-02-26-run-sceadsyr"


def _inclusive_range(*args):
    """
    Implements range with inclusive endpoint, i.e., in the interval [,] instead of [,).
    """
    args = list(args)
    args[0 if len(args) == 1 else 1] += 1
    return range(*args)


def _make_inclusive(spec: slice):
    return slice(spec.start, spec.stop + 1 if spec.stop != -1 else None, spec.step)


def _indices(ts):
    # This function might need to change if the implementation details of
    # TimeStepSpec ever change.
    # It also relies on the picmi object and the pypicongpu object using
    # the same internal variable and storage layout.
    return sorted(reduce(set.union, (list(_inclusive_range(STEPS))[_make_inclusive(spec)] for spec in ts.specs), set()))


def setup_sim():
    sim = basic_simulation()
    for laser in LASERS:
        sim.add_laser(laser, None)
    sim.diagnostics = [Checkpoint(TimeStepSpec[::100])]
    if RUN_DIR:
        sim.picongpu_get_runner().run_dir = RUN_DIR
    else:
        sim.step(STEPS)
    return sim


SIM = None


class TestLasers(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
        self.sim = SIM
        self.coordinates = np.transpose(
            np.meshgrid(
                *(
                    np.linspace(low, up, n, endpoint=False)
                    for low, up, n in zip(LOWER_BOUNDARY, UPPER_BOUNDARY, NUMBER_OF_CELLS)
                )
            ),
            (0, 2, 1, 3),
        )
        self.checkpoint_steps = _indices(
            self.sim.diagnostics[0].period.get_as_pypicongpu(self.sim.time_step_size, self.sim.max_steps)
        )

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

        # def test_grid(self):
        #    np.testing.assert_allclose(
        #        read_grids(self.result_path / "simOutput" / "checkpoints" / "checkpoint_%T.bp5")["E"], self.coordinates
        #    )

    def test_total_E_field(self):
        errors = []
        # for it in self.checkpoint_steps:
        for it in [300, 400]:
            expected = np.sum([laser.E(*self.coordinates, t=it * self.sim.time_step_size) for laser in LASERS], axis=0)
            fields = read_fields(self.result_path / "simOutput" / "checkpoints" / "checkpoint_%T.bp5", iteration=it)
            try:
                np.testing.assert_allclose(fields["E"], expected)
            except AssertionError as error:
                print(f"Failed at {it=}.")
                errors.append(error)
                # raise AssertionError(f"Assertion failed at {it=}. See above.") from error
            plot_half_box_slices(self.coordinates, fields["E"], title=f"Found at {it=}")
            plot_half_box_slices(self.coordinates, expected, title=f"Expected at {it=}")
            plot_half_box_slices(self.coordinates, expected / fields["E"], title=f"Expected/found at {it=}")
            plot_half_box_slices(self.coordinates, fields["E"] - expected, title=f"Found - expected at {it=}")
            plt.figure()
            values = np.abs(expected / fields["E"]).reshape(-1)
            print(np.sum(values > 1.0e-10), np.sum(expected == 0), values.shape)
            log_values = np.log(values[values > 1.0e-10])
            print(values)
            print(log_values)
            breakpoint()
            plt.hist(log_values, bins=1000)
            plt.show()
        for error in errors:
            print(error)
            print(*error.args, sep="\n")
            print()

        plt.show()
        raise errors[-1]


def plot(*data, title, unified_cb=False):
    vmin_vmax = {"vmin": min(*map(np.min, data)), "vmax": max(*map(np.max, data))} if unified_cb else {}

    fig, ax = plt.subplots(len(data))
    fig.suptitle(title)
    for a, d in zip(ax, data):
        im = a.imshow(np.log(d[0, NUMBER_OF_CELLS[0] // 2, ...]), **vmin_vmax, origin="lower")
        if not unified_cb:
            fig.colorbar(im)
    if unified_cb:
        fig.colorbar(im)


def plot_half_box_slices(grid, data, field_components="xyz", title=""):
    fig, ax = plt.subplots(len(field_components), 3, squeeze=False)
    fig.suptitle(title)
    for i, c in enumerate("xyz"):
        ax_index = 0
        for field_component, d in zip("xyz", data):
            if field_component in field_components:
                a = ax[ax_index, i]
                ax_index += 1
                s = np.roll([d.shape[i] // 2, slice(None), slice(None)], shift=i)
                coordinates = sorted(set(range(3)) - {i})
                x, y = grid[coordinates, *s].reshape(2, -1)
                z = d[*s].reshape(-1)
                im = a.scatter(x, y, c=z)
                fig.colorbar(im)
                a.set_title(f"E{field_component}, slice: {c}=Box/2")
                a.set_xlabel("xyz"[coordinates[0]])
                a.set_ylabel("xyz"[coordinates[1]])
    fig.tight_layout()


def plot_z_slices(grid, data, field_components="xyz", title=""):
    num_slices = 4
    fig, ax = plt.subplots(len(field_components), num_slices, squeeze=False)
    fig.suptitle(title)
    vmin = data.min()
    vmax = data.max()
    for i, z_slice in enumerate(range(0, data.shape[1], data.shape[1] // num_slices)):
        ax_index = 0
        for field_component, d in zip("xyz", data):
            if field_component in field_components:
                a = ax[ax_index, i]
                ax_index += 1
                x, y = grid[[0, 1], :, :, z_slice].reshape(2, -1)
                z = d[:, :, z_slice].reshape(-1)
                im = a.scatter(x, y, c=z, vmin=vmin, vmax=vmax)
                fig.colorbar(im)
                a.set_title(f"E{field_component}, slice: z={z_slice}")
                a.set_xlabel("x")
                a.set_ylabel("y")
    fig.tight_layout()
