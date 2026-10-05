"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Full end-to-end test of the standard momentum surface of
:class:`~picongpu.picmi.distribution.AnalyticDistribution`:
the constant ``momentum_expressions`` (a ``gamma * velocity`` drift) and
``momentum_spread_expressions`` (a Gaussian thermal sigma) -- together with the
``user_defined_kw`` substitution feeding them -- must actually reach the
particle species after the input file is compiled and run.

The drift is checked exactly (it assigns the same momentum to every
macroparticle), the spread statistically (an anisotropic 3:1 sigma ratio is
robust against the random draw and independent of weighting and units).
"""

import logging
from pathlib import Path
from unittest import TestCase

import numpy as np
from picongpu import rc_params
from picongpu.picmi import (
    AnalyticDistribution,
    Cartesian3DGrid,
    ElectromagneticSolver,
    PseudoRandomLayout,
    Simulation,
    Species,
)
from picongpu.picmi.diagnostics import Checkpoint, TS

from .arbitrary_parameters import NUMBER_OF_CELLS, UPPER_BOUNDARY, directory_in_home, gather_results
from .compare_particles import read_particles

logging.basicConfig(level=logging.INFO)

# gamma * velocity along y [m/s]; the user_defined_kw ("vdrift") is substituted in
GAMMA_VELOCITY_Y = 3.0e7
# anisotropic Gaussian thermal sigma [m/s] on x and y; z has no spread
SPREAD_X = 1.0e5
SPREAD_Y = 3.0e5

DRIFT_DISTRIBUTION = AnalyticDistribution(
    density_expression="n0",
    n0=1.0e25,
    momentum_expressions=[None, "vdrift", None],
    vdrift=GAMMA_VELOCITY_Y,
)

SPREAD_DISTRIBUTION = AnalyticDistribution(
    density_expression="n0",
    n0=1.0e25,
    momentum_spread_expressions=["vthx", "vthy", None],
    vthx=SPREAD_X,
    vthy=SPREAD_Y,
)

# the same surface, but with the per-axis callable forms and their keyword
# substitution instead of the string expressions
FUNCTION_DISTRIBUTION = AnalyticDistribution(
    density_expression="n0",
    n0=1.0e25,
    momentum_functions=[None, lambda x, y, z, vdrift: vdrift, None],
    momentum_spread_functions=[lambda x, y, z, vthx: vthx, lambda x, y, z, vthy: vthy, None],
    vdrift=GAMMA_VELOCITY_Y,
    vthx=SPREAD_X,
    vthy=SPREAD_Y,
)


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


def setup_sim():
    sim = basic_simulation()
    sim.add_species(
        Species(name="drift", particle_type="electron", initial_distribution=DRIFT_DISTRIBUTION),
        PseudoRandomLayout(n_macroparticles_per_cell=2),
    )
    sim.add_species(
        Species(name="spread", particle_type="electron", initial_distribution=SPREAD_DISTRIBUTION),
        PseudoRandomLayout(n_macroparticles_per_cell=2),
    )
    sim.add_species(
        Species(name="function", particle_type="electron", initial_distribution=FUNCTION_DISTRIBUTION),
        PseudoRandomLayout(n_macroparticles_per_cell=2),
    )
    sim.diagnostics = [Checkpoint(period=TS[:])]
    if "rosi-hzdr" in rc_params.get("preset", "bash"):
        # On ROSI, the tmp directories are inaccessible to compute nodes.
        sim.picongpu_get_runner().setup_dir = directory_in_home() / "setup"
        sim.picongpu_get_runner().run_dir = directory_in_home() / "run"
    # Two species inflate the per-species translation units; keep the build small.
    sim.step(0, jobs=2)
    return sim


SIM = None


class TestAnalyticDistributionMomentum(TestCase):
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
    def particles(self):
        return read_particles(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5")

    def test_momentum_drift_reaches_species(self):
        # a constant gamma*velocity along y assigns exactly p_y and leaves x and z at zero
        particles = self.particles.loc(axis=0)["drift"]
        momenta = particles[["momentum_x", "momentum_y", "momentum_z"]].to_numpy()
        assert len(momenta) > 0
        np.testing.assert_allclose(momenta[:, 0], 0.0, atol=0.0)
        np.testing.assert_allclose(momenta[:, 2], 0.0, atol=0.0)
        # every macroparticle gets the very same drift momentum
        np.testing.assert_allclose(momenta[:, 1], momenta[0, 1], rtol=1.0e-6)
        assert abs(momenta[0, 1]) > 0.0

    def test_momentum_spread_reaches_species(self):
        # anisotropic Gaussian sigmas are visible as a 3:1 standard-deviation
        # ratio (independent of weighting and units); z must stay exactly zero.
        particles = self.particles.loc(axis=0)["spread"]
        momenta = particles[["momentum_x", "momentum_y", "momentum_z"]].to_numpy()
        assert len(momenta) > 1
        std_x = momenta[:, 0].std()
        std_y = momenta[:, 1].std()
        assert std_x > 0.0
        assert std_y > 0.0
        np.testing.assert_allclose(momenta[:, 2], 0.0, atol=0.0)
        np.testing.assert_allclose(std_y / std_x, SPREAD_Y / SPREAD_X, rtol=0.05)
        # the thermal spread is centred: the mean is small on the sigma scale
        np.testing.assert_allclose(momenta[:, :2].mean(axis=0), 0.0, atol=0.05 * std_x)

    def test_momentum_function_reaches_species(self):
        # the per-axis callable forms (with keyword-substituted drift and spread)
        # must reach the particle species exactly like the string expressions
        drift = self.particles.loc(axis=0)["function"]
        momenta = drift[["momentum_x", "momentum_y", "momentum_z"]].to_numpy()
        assert len(momenta) > 0
        # z carries neither a momentum nor a spread expression, so it stays exactly
        # zero; x carries the thermal spread and y the drift plus the (three times
        # broader) spread.
        np.testing.assert_allclose(momenta[:, 2], 0.0, atol=0.0)
        std_x = momenta[:, 0].std()
        std_y = momenta[:, 1].std()
        assert std_x > 0.0
        assert std_y > 0.0
        np.testing.assert_allclose(std_y / std_x, SPREAD_Y / SPREAD_X, rtol=0.05)
