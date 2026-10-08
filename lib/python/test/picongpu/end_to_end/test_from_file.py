"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

End-to-end test for loading a species from an external openPMD particle file
(https://github.com/chillenzer-agents/picongpu/issues/176).

A reference particle set is written to an openPMD file, a 0-step simulation
declares the species through ``FromFileDistribution``, and the dumped particles
are required to be identical to the input (up to floating-point accuracy). The
setup mirrors the other distribution end-to-end tests so that interactions with
the ordinary initialisation paths are exercised as well.
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
    FromFileDistribution,
    Simulation,
    Species,
)
from picongpu.picmi.diagnostics import Checkpoint, TS

from .arbitrary_parameters import CELL_SIZE, NUMBER_OF_CELLS, UPPER_BOUNDARY, directory_in_home, gather_results
from .compare_particles import read_particles

logging.basicConfig(level=logging.INFO)

PARTICLE_SHAPE = "other:counter"
# no underscore: compare_particles splits species names on the first "_"
SPECIES_NAME = "bunch"
NUMBER_OF_PARTICLES = 100


def _reference_particles():
    """Deterministic global positions, momenta and weights in SI units."""
    rng = np.random.default_rng(20260101)
    cell_size = np.asarray(CELL_SIZE, dtype=np.float64)
    shape = np.asarray(NUMBER_OF_CELLS, dtype=np.int64)
    # global cell indices, avoiding the outer cells
    cells = np.stack([rng.integers(1, shape[i] - 1, size=NUMBER_OF_PARTICLES) for i in range(3)], axis=1)
    # in-cell fractional position
    in_cell = rng.uniform(0.0, 1.0, size=(NUMBER_OF_PARTICLES, 3))
    # cell-beginning position in SI
    cell_beginning = cells * cell_size
    # the physical position relative to the cell beginning is the fraction times
    # the cell size; PIConGPU's openPMD `position` carries the in-cell fraction
    # with `unitSI` = cell size
    position = cell_beginning + in_cell * cell_size
    momentum = rng.normal(0.0, 1.0e-23, size=(NUMBER_OF_PARTICLES, 3))
    weighting = rng.uniform(1.0e5, 2.0e5, size=NUMBER_OF_PARTICLES)
    return {
        "in_cell": in_cell,
        "cell_beginning": cell_beginning,
        "position": position,
        "momentum": momentum,
        "weighting": weighting,
    }


def _write_reference_file(path: Path):
    """Write an openPMD file in PIConGPU's in-cell ``position``/``positionOffset`` form.

    ``position`` is the in-cell fraction with ``unitSI`` = cell size, so its SI
    value is the distance from the cell beginning; ``positionOffset`` is the
    cell-beginning position in SI. No ``particlePatches`` are written, so this
    exercises the general (standard-compliant) loader path.
    """
    reference = _reference_particles()
    series = opmd.Series(str(path), opmd.Access.create)
    species = series.iterations[0].particles[SPECIES_NAME]

    position = species["position"]
    position_offset = species["positionOffset"]
    momentum = species["momentum"]
    weighting = species["weighting"]

    double = opmd.determine_datatype(np.dtype("float64"))
    for i, axis in enumerate(("x", "y", "z")):
        position[axis].reset_dataset(opmd.Dataset(double, (NUMBER_OF_PARTICLES,)))
        position[axis].unit_SI = CELL_SIZE[i]
        # ``[:, i]`` is a strided column view of a C-order array, which the
        # openPMD C backend rejects; store a row-major contiguous copy.
        position[axis].store_chunk(np.ascontiguousarray(reference["in_cell"][:, i]))

        position_offset[axis].reset_dataset(opmd.Dataset(double, (NUMBER_OF_PARTICLES,)))
        position_offset[axis].unit_SI = 1.0
        position_offset[axis].store_chunk(np.ascontiguousarray(reference["cell_beginning"][:, i]))

        momentum[axis].reset_dataset(opmd.Dataset(double, (NUMBER_OF_PARTICLES,)))
        momentum[axis].unit_SI = 1.0
        momentum[axis].store_chunk(np.ascontiguousarray(reference["momentum"][:, i]))

    scalar = opmd.Mesh_Record_Component.SCALAR
    weighting[scalar].reset_dataset(opmd.Dataset(double, (NUMBER_OF_PARTICLES,)))
    weighting[scalar].unit_SI = 1.0
    weighting[scalar].store_chunk(np.ascontiguousarray(reference["weighting"]))

    series.flush()
    series.close()
    return reference


def basic_simulation():
    return Simulation(
        max_steps=0,
        solver=ElectromagneticSolver(
            method="Yee",
            cfl=1.0,
            grid=Cartesian3DGrid(
                number_of_cells=NUMBER_OF_CELLS,
                lower_bound=[0, 0, 0],
                upper_bound=UPPER_BOUNDARY,
                lower_boundary_conditions=["open", "open", "open"],
                upper_boundary_conditions=["open", "open", "open"],
            ),
        ),
    )


RUN_DIR = ""


def setup_sim():
    sim = basic_simulation()
    input_dir = Path(sim.picongpu_get_runner().setup_dir).parent / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    input_file = input_dir / "from_file_bunch.bp5"
    if not input_file.exists():
        _write_reference_file(input_file)
    species = Species(
        name=SPECIES_NAME,
        particle_type="electron",
        initial_distribution=FromFileDistribution(file_path=str(input_file), iteration=0),
        particle_shape=PARTICLE_SHAPE,
    )
    sim.add_species(species, None)
    sim.diagnostics = [Checkpoint(period=TS[:])]
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


class TestFromFile(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
        self.sim = SIM
        gather_results(self.result_path)
        # read_particles applies each record's unitSI, so the values are SI.
        self._particles = read_particles(self._dump_path()).loc(axis=0)[SPECIES_NAME].reset_index(drop=True)

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

    def _dump_path(self):
        return self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5"

    def test_particle_count(self):
        assert len(self._particles) == NUMBER_OF_PARTICLES

    def test_global_positions_are_reproduced(self):
        reference = _reference_particles()
        global_position = (
            self._particles[["position_x", "position_y", "position_z"]].to_numpy()
            + self._particles[["positionOffset_x", "positionOffset_y", "positionOffset_z"]].to_numpy()
        )
        assert np.allclose(global_position, reference["position"], rtol=1.0e-6, atol=1.0e-12)

    def test_momenta_are_reproduced(self):
        reference = _reference_particles()
        momentum = self._particles[["momentum_x", "momentum_y", "momentum_z"]].to_numpy()
        assert np.allclose(momentum, reference["momentum"], rtol=1.0e-6, atol=1.0e-30)

    def test_weightings_are_reproduced(self):
        reference = _reference_particles()
        weighting = self._particles["weighting"].to_numpy()
        assert np.allclose(weighting, reference["weighting"], rtol=1.0e-6, atol=1.0e-12)
