"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
from pathlib import Path
from unittest import TestCase, main

import numpy as np
from .arbitrary_parameters import (
    NUMBER_OF_CELLS,
    UPPER_BOUNDARY,
)
from .compare_particles import apply_range, load_diagnostic_result, read_particles, sort_particles, read_fields
from .distributions import Gaussian, SphereFlanks
from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    GriddedLayout,
    Simulation,
    Species as Species,
)
from picongpu.picmi.diagnostics import (
    Checkpoint,
    FieldDump,
    OpenPMDConfig,
    ParticleDump,
    TimeStepSpec,
)

logging.basicConfig(level=logging.INFO)

LAYOUT = GriddedLayout(n_macroparticles_per_cell=2)
SPECIES = [
    Species(
        name="Gaussian_predefined",
        particle_type="electron",
        initial_distribution=Gaussian().distributions["predefined"],
    ),
    Species(
        name="SphereFlanks_free_form",
        particle_type="electron",
        initial_distribution=SphereFlanks().distributions["free_form"],
    ),
]


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


def generate_diagnostics(species):
    options = OpenPMDConfig(
        file="other_name", ext=".h5", infix="", data_preparation_strategy="doubleBuffer", range=[17, (25, 40), None]
    )
    particles = [ParticleDump(species=s) for s in species] + [
        ParticleDump(species=species[0], options=options),
    ]
    native_fields = [FieldDump(fieldname=fieldname) for fieldname in ["E", "B"]]
    derived_fields = []
    return particles + native_fields + derived_fields


RUN_DIR = ""


def setup_sim():
    sim = basic_simulation()
    for species in SPECIES:
        sim.add_species(species, LAYOUT)
    sim.diagnostics = [Checkpoint(TimeStepSpec[:])] + generate_diagnostics(SPECIES)
    if RUN_DIR:
        sim.picongpu_get_runner().run_dir = RUN_DIR
    else:
        sim.step(0)
    return sim


SIM = None


class TestDiagnostics(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
        self.sim = SIM

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

    def test_particle_dump(self):
        for diag in self.sim.diagnostics:
            if isinstance(diag, ParticleDump):
                from_checkpoint = sort_particles(
                    apply_range(
                        read_particles(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5"),
                        diag.options.range,
                    )
                ).loc(axis=0)[*diag.species.name.split("_", maxsplit=1)]
                from_diagnostics = sort_particles(load_diagnostic_result(diag, self.result_path))
                np.testing.assert_allclose(from_checkpoint, from_diagnostics)

    def test_field_dump(self):
        for diag in self.sim.diagnostics:
            if isinstance(diag, FieldDump):
                np.testing.assert_allclose(
                    load_diagnostic_result(diag, self.result_path),
                    read_fields(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5")[
                        diag.fieldname
                    ],
                )


if __name__ == "__main__":
    main()
