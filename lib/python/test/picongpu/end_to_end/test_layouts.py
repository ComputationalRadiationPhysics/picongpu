"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import logging
from pathlib import Path
from unittest import TestCase

import numpy as np
import pandas as pd
from picongpu import rc_params
from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    GriddedLayout,
    OnePositionLayout,
    PseudoRandomLayout,
    Simulation,
)
from picongpu.picmi import Species as Species
from picongpu.picmi.diagnostics import (
    Checkpoint,
    TimeStepSpec,
)

from .arbitrary_parameters import CELL_SIZE, NUMBER_OF_CELLS, UPPER_BOUNDARY, directory_in_home, gather_results
from .compare_particles import (
    read_particles,
    sort_particles,
)
from .distributions import Uniform

logging.basicConfig(level=logging.INFO)

PARTICLE_SHAPE = "counter"
LAYOUTS = {
    "oneposition_1": OnePositionLayout(n_macroparticles_per_cell=1),
    "oneposition_2": OnePositionLayout(n_macroparticles_per_cell=2),
    "oneposition_2_offset": OnePositionLayout(n_macroparticles_per_cell=2, in_cell_offset=(0.1, 0.2, 0.3)),
    "gridded_1": GriddedLayout(n_macroparticle_per_cell=(1, 1, 1)),
    "gridded_1_2_3": GriddedLayout(n_macroparticle_per_cell=(1, 2, 3)),
    "pseudorandom_1": PseudoRandomLayout(n_macroparticles_per_cell=1),
    "pseudorandom_2": PseudoRandomLayout(n_macroparticles_per_cell=2),
}


def generate_species(layout_name):
    return Species(
        name=f"{layout_name}",
        particle_type="electron",
        initial_distribution=Uniform().distributions["predefined"],
        particle_shape=PARTICLE_SHAPE,
    )


SPECIES_AND_LAYOUTS = {name: (generate_species(name), layout) for name, layout in LAYOUTS.items()}


def basic_simulation():
    return Simulation(
        max_steps=0,
        solver=ElectromagneticSolver(
            method="Yee",
            cfl=1.0,
            grid=Cartesian3DGrid(
                number_of_cells=NUMBER_OF_CELLS,
                lower_bound=[0, 0, 0],
                # cell size is slightly different from 1
                upper_bound=UPPER_BOUNDARY,
                lower_boundary_conditions=["open", "open", "open"],
                upper_boundary_conditions=["open", "open", "open"],
            ),
        ),
    )


RUN_DIR = ""


def setup_sim():
    sim = basic_simulation()
    for species, layout in SPECIES_AND_LAYOUTS.values():
        sim.add_species(species, layout)
    sim.diagnostics = [Checkpoint(TimeStepSpec[:])]
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


class TestLayouts(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
            self.sim = SIM
            gather_results(self.result_path)
        self.sim = SIM
        self.index = ["layout", "parameters"]
        self.offset_names = ["positionOffset_x", "positionOffset_y", "positionOffset_z"]
        self.position_names = ["position_x", "position_y", "position_z"]
        self.particles = (
            sort_particles(read_particles(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5"))
            .rename_axis(index=dict(zip(["setup", "impl"], self.index)))
            .reset_index((0, 1), drop=True)
        )

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

    def test_total_particle_count(self):
        particle_count = (
            pd.Series(
                {
                    tuple(name.split("_", maxsplit=1)): np.prod(NUMBER_OF_CELLS)
                    * np.asarray(layout.n_macroparticles_per_cell, dtype=int).prod()
                    for name, (_, layout) in SPECIES_AND_LAYOUTS.items()
                },
                name="expected",
            )
            .rename_axis(index=self.index)
            .to_frame()
        )
        particle_count["found"] = self.particles["weighting"].groupby(level=self.index).count()
        pd.testing.assert_series_equal(particle_count["found"], particle_count["expected"], check_names=False)

    def test_particle_count_per_cell(self):
        particle_count = (
            pd.Series(
                {
                    tuple(name.split("_", maxsplit=1)): np.asarray(layout.n_macroparticles_per_cell, dtype=int).prod()
                    for name, (_, layout) in SPECIES_AND_LAYOUTS.items()
                },
                name="expected",
            )
            .rename_axis(index=self.index)
            .to_frame()
        )
        # Our layouts all have the exact same number of macroparticles per cell by now.
        # So, the `drop_duplicates` will leave exactly one result
        # and this result can be compared to our expected particle count.
        # If there is a wrong result, the assertion below will blow up.
        # If there are multiple results, something will blow up as well
        # either the assertion below or the assignment here already.
        # It will definitely fail but don't be confused, if it's already in the assignment.
        particle_count["found"] = (
            self.particles.reset_index(drop=False)
            .groupby(by=[*self.index, *self.offset_names])["weighting"]
            .count()
            .reset_index(self.offset_names, drop=True)
            .reset_index(drop=False)
            .drop_duplicates()
            .set_index(self.index)
        )
        pd.testing.assert_series_equal(particle_count["found"], particle_count["expected"], check_names=False)

    def test_one_position_has_correct_in_cell_offset(self):
        offsets = pd.concat(
            {
                "expected": pd.DataFrame(
                    {
                        tuple(name.split("_", maxsplit=1)): np.asarray(layout.in_cell_offset) * CELL_SIZE
                        for name, (_, layout) in SPECIES_AND_LAYOUTS.items()
                        if name.startswith("oneposition")
                    },
                    index=self.position_names,
                ).T,
                "found": self.particles.loc()[("oneposition", slice(None)), self.position_names],
            },
            axis=1,
        )
        pd.testing.assert_frame_equal(offsets["expected"], offsets["found"], check_dtype=False)

    def test_gridded_has_correct_in_cell_offsets(self):
        offsets = pd.concat(
            {
                "expected": pd.concat(
                    {
                        tuple(name.split("_", maxsplit=1)): pd.DataFrame.from_records(
                            data=layout.in_cell_offsets * np.reshape(CELL_SIZE, (1, -1)),
                            columns=self.position_names,
                        )
                        for name, (_, layout) in SPECIES_AND_LAYOUTS.items()
                        if name.startswith("gridded")
                    },
                    axis=0,
                )
                .reset_index(level=-1, drop=True)
                .rename_axis(index=self.index),
                "found": self.particles.loc()[("gridded", slice(None)), self.position_names]
                .reset_index(drop=False)
                .drop_duplicates()
                .set_index(self.index),
            },
            axis=1,
        )
        pd.testing.assert_frame_equal(offsets["expected"], offsets["found"], check_dtype=False)

    def test_pseudorandom_in_cell_offsets_are_all_unique(self):
        offsets = self.particles.loc()[("pseudorandom", slice(None)), self.position_names]
        pd.testing.assert_frame_equal(offsets.drop_duplicates(), offsets)
