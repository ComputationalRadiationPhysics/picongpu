"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
import unittest
from pathlib import Path

import numpy as np
from picongpu import picmi
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec

from .arbitrary_parameters import (
    CELL_SIZE,
    NUMBER_OF_CELLS,
    UPPER_BOUNDARY,
)
from .binning_functors import binning_diagnostics
from .compare_particles import (
    compare_particles,
    read_binning,
    read_densities_into_mesh,
    read_particles,
    read_position_check,
)
from .distributions import DISTRIBUTIONS

logging.basicConfig(level=logging.INFO)

LAYOUT = picmi.GriddedLayout(n_macroparticles_per_cell=2)


def basic_simulation():
    return picmi.Simulation(
        max_steps=0,
        solver=picmi.ElectromagneticSolver(
            method="Yee",
            cfl=1.0,
            grid=picmi.Cartesian3DGrid(
                number_of_cells=NUMBER_OF_CELLS,
                lower_bound=[0, 0, 0],
                # cell size is slightly different from 1
                upper_bound=UPPER_BOUNDARY,
                lower_boundary_conditions=["open", "open", "open"],
                upper_boundary_conditions=["open", "open", "open"],
            ),
        ),
    )


def generate_name(name, suffix):
    return name + "_" + suffix


def generate_species(name, distribution):
    return [
        picmi.Species(
            name=name,
            particle_type="H",
            initial_distribution=distribution,
            picongpu_fixed_charge=True,
        )
    ]


def setup_sim():
    sim = basic_simulation()

    species = sum(
        (
            generate_species(generate_name(name, suffix), dist)
            for name, distributions in DISTRIBUTIONS.items()
            for suffix, dist in distributions.items()
        ),
        [],
    )

    diagnostics = [picmi.diagnostics.Checkpoint(TimeStepSpec[:])] + binning_diagnostics(species, sim.time_step_size)

    for s in species:
        sim.add_species(s, LAYOUT)
    sim.diagnostics = diagnostics

    sim.step(0)
    return sim


# only run this once, so we don't compile each and every time
SIM = None


class TestFreeFormulaDensity(unittest.TestCase):
    _result_path = None

    def setUp(self):
        if self._result_path is None:
            global SIM
            if SIM is None:
                SIM = setup_sim()
            self.sim = SIM
        else:
            for d in DISTRIBUTIONS:
                for f in DISTRIBUTIONS[d].values():
                    f.cell_size = CELL_SIZE

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim._Simulation__runner.run_dir)
        return self._result_path

    def test_compare_particles_pairwise(self):
        self.assertTrue(compare_particles(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5"))

    def test_compare_particles_against_call_operator(self):
        particles = read_densities_into_mesh(
            self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5",
            NUMBER_OF_CELLS,
            CELL_SIZE,
        )
        for setup, dists in DISTRIBUTIONS.items():
            for impl, func in dists.items():
                density = particles.loc(axis=0)[setup][impl]
                values = func(*(np.indices(density.shape) + 0.5) * np.reshape(CELL_SIZE, (3, 1, 1, 1)))
                np.testing.assert_allclose(density, values, rtol=1.0e-4)

    def test_compare_binning_against_call_operator(self):
        for setup, dists in DISTRIBUTIONS.items():
            for impl, func in dists.items():
                mesh = read_binning(
                    self.result_path
                    / "simOutput"
                    / "binningOpenPMD"
                    / f"particleDensity_{generate_name(setup, impl)}_000000.bp5",
                    CELL_SIZE,
                )
                values = func(*(np.indices(mesh.shape) + 0.5) * np.reshape(CELL_SIZE, (3, 1, 1, 1)))

                np.testing.assert_allclose(mesh, values, rtol=1.0e-4)

    def test_compare_binning_against_particle_density(self):
        particles = read_densities_into_mesh(
            self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5",
            NUMBER_OF_CELLS,
            CELL_SIZE,
        )

        for setup, dists in DISTRIBUTIONS.items():
            for impl in dists.keys():
                mesh = read_binning(
                    self.result_path
                    / "simOutput"
                    / "binningOpenPMD"
                    / f"particleDensity_{generate_name(setup, impl)}_000000.bp5",
                    CELL_SIZE,
                )
                np.testing.assert_allclose(mesh, particles.loc(axis=0)[setup, impl])

    def test_origins_are_all_the_same(self):
        # This test is not very meaningful,
        # only in the way that it compiles and doesn't yield completely bogus results.
        # The different variations of position origins only refer to different things
        # if we have multiple ranks and moving window activated.
        # As we don't in this test, there isn't really much to test
        # except for the fact that all the different coordinates must be identical.
        # The position_check functor does so by counting the correct ones.
        number_of_particles = (
            read_particles(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5")["weighting"]
            .groupby(["setup", "impl"])
            .count()
        )

        for (setup, impl), count in number_of_particles.items():
            self.assertEqual(
                round(
                    read_position_check(
                        self.result_path
                        / "simOutput"
                        / "binningOpenPMD"
                        / f"origin_{generate_name(setup, impl)}_000000.h5"
                    )
                ),
                count,
            )

    def test_unit_conversions_by_hand(self):
        number_of_particles = (
            read_particles(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5")["weighting"]
            .groupby(["setup", "impl"])
            .count()
        )

        for (setup, impl), count in number_of_particles.items():
            self.assertEqual(
                round(
                    read_position_check(
                        self.result_path
                        / "simOutput"
                        / "binningOpenPMD"
                        / f"unit_{generate_name(setup, impl)}_000000.h5"
                    )
                ),
                count,
            )


if __name__ == "__main__":
    unittest.main()
