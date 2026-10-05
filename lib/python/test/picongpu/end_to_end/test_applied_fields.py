"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Full end-to-end test of the applied (background) fields.

Unlike the density distributions, whose effect is visible in the particle
positions/densities, a background field is only applied around the particle
push and is *not* evolved by the field solver. Its most direct observable is
therefore the field dump: the Python layer translates the PICMI applied fields
into the C++ ``fieldBackground.param`` functors, the (compiled) simulation
evaluates them and writes E and B to the checkpoint. This test builds and runs
such a simulation and compares the dumped, unit-converted E/B values against the
call operator of the very same PICMI applied fields.

The simulation is built declaratively in a single ``Simulation(...)`` call; the
setup adds several applied fields (constant, string-expression and callable) so
that the summation into the single C++ background functor pair is covered end to
end.
"""

import logging
from pathlib import Path
from unittest import TestCase

import numpy as np
import openpmd_api as opmd
from picongpu import rc_params
from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    PseudoRandomLayout,
    Simulation,
    Species,
    UniformDistribution,
)
from picongpu.picmi.diagnostics import Checkpoint, TS

from .applied_fields import APPLIED_FIELDS, combined_field_values
from .arbitrary_parameters import CELL_SIZE, NUMBER_OF_CELLS, UPPER_BOUNDARY, directory_in_home, gather_results

logging.basicConfig(level=logging.INFO)

LAYOUT = PseudoRandomLayout(n_macroparticles_per_cell=1)


def basic_simulation():
    """The applied-field simulation, built entirely through the constructor."""
    grid = Cartesian3DGrid(
        number_of_cells=NUMBER_OF_CELLS,
        lower_bound=[0, 0, 0],
        # cell size is slightly different from 1
        upper_bound=UPPER_BOUNDARY,
        lower_boundary_conditions=["open", "open", "open"],
        upper_boundary_conditions=["open", "open", "open"],
    )
    return Simulation(
        max_steps=0,
        solver=ElectromagneticSolver(method="Yee", cfl=1.0, grid=grid),
        species=[Species(particle_type="electron", initial_distribution=UniformDistribution(density=1.0e24))],
        layouts=[LAYOUT],
        applied_fields=APPLIED_FIELDS,
        diagnostics=[Checkpoint(period=TS[:])],
    )


RUN_DIR = ""


def setup_sim():
    sim = basic_simulation()
    if "rosi-hzdr" in rc_params.get("preset", "bash"):
        # On ROSI, the tmp directories are inaccessible to compute nodes.
        sim.picongpu_get_runner().setup_dir = directory_in_home() / "setup"
        sim.picongpu_get_runner().run_dir = directory_in_home() / "run"
    if RUN_DIR:
        sim.picongpu_get_runner().run_dir = RUN_DIR
    else:
        sim.step(0)
    return sim


SIM = None


def _read_field_components(path: Path, field_name: str):
    """
    Read the three components of ``field_name`` and return them in SI units.

    openPMD stores fields as ``F[z][y][x]``; the returned arrays keep that
    layout. Each component is multiplied by its stored ``unit_SI``.
    """
    series = opmd.Series(str(path), opmd.Access.read_only)
    mesh = series.iterations[0].meshes[field_name]
    # ``load_chunk`` only schedules a read; the returned buffers are undefined
    # until ``series.flush()`` runs. Any arithmetic before the flush therefore
    # operates on garbage. Collect the raw float32 chunks first, flush, and only
    # then scale.
    raw = {component: mesh[component].load_chunk() for component in ("x", "y", "z")}
    grid_spacing = np.asarray(mesh.grid_spacing, dtype=np.float64) * mesh.grid_unit_SI
    grid_global_offset = np.asarray(mesh.grid_global_offset, dtype=np.float64) * mesh.grid_unit_SI
    series.flush()
    # Scale in double precision: ``load_chunk`` yields float32 and scaling it by
    # the (large) field ``unit_SI`` overflows float32, which the test session's
    # ``filterwarnings = error`` turns into a failure. The promoted product is
    # the same value with more headroom.
    components = {
        component: raw[component].astype(np.float64) * mesh[component].unit_SI for component in ("x", "y", "z")
    }
    series.close()
    return components, grid_spacing, grid_global_offset


def _cell_centers(shape, grid_global_offset):
    """
    Return the SI coordinates of every cell center, in field layout ``[z][y][x]``.

    The cell size is taken from the input geometry (``CELL_SIZE``) rather than
    from openPMD's ``grid_spacing``: that attribute is stored as float32, which
    perturbs the coordinate by ~1e-7 relative and makes a reference evaluation at
    a sine node (e.g. ``sin(k * x)`` at ``x = 32``) pick up a non-zero value
    instead of the exact zero the C++ core computes in float64. The field layout
    is ``(z, y, x)``, so ``CELL_SIZE`` is reordered accordingly.
    """
    cell_size = (CELL_SIZE[2], CELL_SIZE[1], CELL_SIZE[0])
    z, y, x = (grid_global_offset[axis] + np.arange(shape[axis]) * cell_size[axis] for axis in range(3))
    z, y, x = np.meshgrid(z, y, x, indexing="ij")
    return x, y, z


class TestAppliedFields(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
            self.sim = SIM
            gather_results(self.result_path)
        self.sim = SIM

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

    @property
    def checkpoint(self):
        return self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5"

    def test_dumped_fields_match_the_combined_applied_fields(self):
        time = 0.0  # the step-0 checkpoint is dumped before the first push
        for field_name, prefix, components in (("E", "e", "xyz"), ("B", "b", "xyz")):
            values, _, grid_global_offset = _read_field_components(self.checkpoint, field_name)
            shape = values["x"].shape
            self.assertEqual(tuple(shape), tuple(NUMBER_OF_CELLS[::-1]))
            x, y, z = _cell_centers(shape, grid_global_offset)
            expected = combined_field_values(x, y, z, time)
            for component in components:
                with self.subTest(field=field_name, component=component):
                    reference = expected[f"{prefix.upper()}{component}"]
                    # The dumped field is float32, so its resolution at the field
                    # peak is eps32 * |F|_max. Near a node (e.g. sin(k*x) at
                    # x = pi/k) the reference passes through zero and a pure
                    # relative tolerance is ill-posed: the call operator evaluates
                    # through a lambdified expression whose rounded constants
                    # leave an O(1e-9) residue. Use the dump's own quantization as
                    # the absolute floor; it is far below any non-nodal value.
                    atol = np.finfo(np.float32).eps * np.max(np.abs(reference))
                    np.testing.assert_allclose(values[component], reference, rtol=1.0e-4, atol=atol)

    def test_single_applied_field_component_is_summed(self):
        # Ex is only contributed by the constant field; guard against a silent
        # loss of that (trivial) contribution by comparing against the call
        # operator of that single field.
        values, _, _ = _read_field_components(self.checkpoint, "E")
        np.testing.assert_allclose(values["x"], combined_field_values(0.0, 0.0, 0.0, 0.0)["Ex"], rtol=1.0e-4)
