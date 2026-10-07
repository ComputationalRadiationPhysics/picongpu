"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Full end-to-end test of step-dependent (composite) particle pushers.

A uniform, constant electric field is applied.  A ``Free`` pusher ignores
fields and therefore leaves the (initially zero) momentum untouched, whereas
``Boris`` accelerates the particles.  A species that starts on ``Free`` and
switches to ``Boris`` at a known step must therefore keep its momentum exactly
zero at every dump before the switch and acquire a non-zero momentum after it.

Two species with identical initial conditions make the switch visible in a
single run: ``free`` stays on ``Free`` throughout (momentum stays zero), while
``switched`` uses a :class:`~picongpu.picmi.species.CompositePusher`.
"""

import logging
from pathlib import Path
from unittest import TestCase

import numpy as np
from picongpu import rc_params
from picongpu.picmi import (
    Cartesian3DGrid,
    ConstantAppliedField,
    ElectromagneticSolver,
    PseudoRandomLayout,
    Simulation,
    Species,
    UniformDistribution,
)
from picongpu.picmi.diagnostics import Checkpoint, TS
from picongpu.picmi.species import CompositePusher

from .arbitrary_parameters import NUMBER_OF_CELLS, UPPER_BOUNDARY, directory_in_home, gather_results
from .compare_particles import read_particles

logging.basicConfig(level=logging.INFO)

LAYOUT = PseudoRandomLayout(n_macroparticles_per_cell=1)

# The switch step: `Free` for steps < SWITCH_STEP, `Boris` from SWITCH_STEP.
SWITCH_STEP = 2
MAX_STEPS = 4

# A strong, uniform electric field along x.
E_X = 1.0e10

FREE_SPECIES = Species(
    name="free",
    particle_type="electron",
    initial_distribution=UniformDistribution(density=1.0e24),
    method="free-streaming",
)
SWITCHED_SPECIES = Species(
    name="switched",
    particle_type="electron",
    initial_distribution=UniformDistribution(density=1.0e24),
    method=CompositePusher(
        {
            TS[: SWITCH_STEP - 1]("steps"): "free-streaming",
            TS[SWITCH_STEP:]("steps"): "Boris",
        }
    ),
)


def basic_simulation():
    grid = Cartesian3DGrid(
        number_of_cells=NUMBER_OF_CELLS,
        lower_bound=[0, 0, 0],
        upper_bound=UPPER_BOUNDARY,
        lower_boundary_conditions=["open", "open", "open"],
        upper_boundary_conditions=["open", "open", "open"],
    )
    return Simulation(
        max_steps=MAX_STEPS,
        solver=ElectromagneticSolver(method="Yee", cfl=1.0, grid=grid),
        species=[FREE_SPECIES, SWITCHED_SPECIES],
        layouts=[LAYOUT, LAYOUT],
        applied_fields=[ConstantAppliedField(Ex=E_X)],
        diagnostics=[Checkpoint(period=TS[:]("steps"))],
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
        sim.step(MAX_STEPS, jobs=2)
    return sim


SIM = None


class TestCompositePusher(TestCase):
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

    def checkpoint(self, step):
        return self.result_path / "simOutput" / "checkpoints" / f"checkpoint_{step:06d}.bp5"

    def momenta(self, step, name):
        particles = read_particles(self.checkpoint(step)).loc(axis=0)[name]
        momenta = particles[["momentum_x", "momentum_y", "momentum_z"]].to_numpy()
        assert len(momenta) > 0
        return momenta

    def test_free_pusher_never_accelerates(self):
        # `Free` ignores the electric field: momentum stays exactly zero throughout.
        for step in (0, 1, SWITCH_STEP, MAX_STEPS - 1):
            with self.subTest(step=step):
                np.testing.assert_allclose(self.momenta(step, "free"), 0.0, atol=0.0)

    def test_switched_pusher_is_free_before_and_boris_after(self):
        # before the switch (steps 0, 1) `Free` leaves momentum untouched ...
        for step in (0, 1):
            with self.subTest(step=step):
                np.testing.assert_allclose(self.momenta(step, "switched"), 0.0, atol=0.0)
        # ... and from the switch step onwards `Boris` accelerates along E_x.
        for step in (SWITCH_STEP, MAX_STEPS - 1):
            with self.subTest(step=step):
                momenta = self.momenta(step, "switched")
                assert np.any(np.abs(momenta[:, 0]) > 0.0), "Boris should have accelerated along x after the switch"
                np.testing.assert_allclose(momenta[:, 1:], 0.0, atol=0.0)
