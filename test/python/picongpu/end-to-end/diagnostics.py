"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
from functools import partial
from pathlib import Path
from unittest import TestCase, main

import numpy as np
from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    GriddedLayout,
    Simulation,
)
from picongpu.picmi import Species as Species
from picongpu.picmi.diagnostics import (
    Binning,
    BinningAxis,
    BinSpec,
    Checkpoint,
    DerivedFieldDump,
    NativeFieldDump,
    OpenPMDConfig,
    ParticleDump,
    ParticleFunctor,
    TimeStepSpec,
)
from sympy import Piecewise

from .arbitrary_parameters import (
    CELL_SIZE,
    NUMBER_OF_CELLS,
    UPPER_BOUNDARY,
)
from .compare_particles import (
    apply_range,
    load_diagnostic_result,
    read_fields,
    read_particles,
    sort_particles,
)
from .distributions import Gaussian, SphereFlanks

logging.basicConfig(level=logging.INFO)

LAYOUT = GriddedLayout(n_macroparticles_per_cell=2)
PARTICLE_SHAPE = "counter"
SPECIES = [
    Species(
        name="Gaussian_predefined",
        particle_type="electron",
        initial_distribution=Gaussian().distributions["predefined"],
        particle_shape=PARTICLE_SHAPE,
    ),
    Species(
        name="SphereFlanks_free_form",
        particle_type="electron",
        initial_distribution=SphereFlanks().distributions["free_form"],
        particle_shape=PARTICLE_SHAPE,
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


CUTOFF_ENERGY = 10.0
FUNCTORS = [
    # Currently no eligible particles available:
    # ParticleFunctor(name="bound_electrons", functor=lambda p: p.get("boundElectrons")),
    # Somehow off by some factor:
    # ParticleFunctor(name="charge_density", functor=lambda p: p.get("charge") / np.prod(CELL_SIZE)),
    ParticleFunctor(name="particle_counter", functor=lambda p: p.get("weighting")),
    ParticleFunctor(name="density", functor=lambda p: p.get("weighting") / np.prod(CELL_SIZE)),
    ParticleFunctor(name="kinetic_energy", functor=lambda p: p.get("kinetic energy")),
    ParticleFunctor(name="kinetic_energy_density", functor=lambda p: p.get("kinetic energy") / np.prod(CELL_SIZE)),
    ParticleFunctor(
        name="kinetic_energy_density_cutoff",
        functor=lambda p: Piecewise(
            (
                p.get("kinetic energy") / np.prod(CELL_SIZE),
                p.get("kinetic energy") < CUTOFF_ENERGY * p.get("weighting"),
            ),
            (0.0, True),
        ),
    ),
    # Currently no eligible particles available:
    # ParticleFunctor(name="larmor_power", functor=larmor_power),
    ParticleFunctor(name="macroparticle_counter", functor=lambda _: 1, return_type=int),
    # Somehow off by some factor:
    ParticleFunctor(
        name="mid_current_density_x",
        functor=lambda p: p.get("charge")
        / np.prod(CELL_SIZE)
        * p.get("momentum")[0]
        / (p.get("gamma") * p.get("mass")),
    ),
    ParticleFunctor(name="momentum_y", functor=lambda p: p.get("momentum")[1]),
    # Duplicated just to test what happens:
    # ParticleFunctor(name="momentum_y", functor=lambda p: p.get("momentum")[1]),
    ParticleFunctor(name="momentum_density_z", functor=lambda p: p.get("momentum")[2] / np.prod(CELL_SIZE)),
    ParticleFunctor(name="weighted_velocity_x", functor=lambda p: p.get("velocity")[0] * p.get("weighting")),
]


def generate_particle_dumps(species):
    options = OpenPMDConfig(
        file="other_name", ext=".h5", infix="", data_preparation_strategy="doubleBuffer", range=[17, (25, 40), None]
    )
    return [ParticleDump(species=s) for s in species] + [ParticleDump(species=species[0], options=options)]


def generate_native_field_dumps():
    return [NativeFieldDump(fieldname=fieldname) for fieldname in ["E", "B"]]


def generate_derived_field_dumps(species, functors):
    return [DerivedFieldDump(species=s, functor=f) for s in species for f in functors]


def position(particle, i):
    return particle.get("position", origin="total", precision="sub_cell", unit="si")[i] // CELL_SIZE[i]


POSITION_AXES = [
    BinningAxis(
        ParticleFunctor(
            # We prefer `partial` over lambda functions in this situation
            # because of lambda's late binding.
            name=f"position{i}",
            functor=partial(position, i=i),
            return_type=int,
        ),
        BinSpec("linear", 0, NUMBER_OF_CELLS[i], NUMBER_OF_CELLS[i]),
        use_overflow_bins=False,
    )
    for i in range(3)
]


def generate_derived_field_dumps_as_binnings(species, functors):
    return [
        Binning(name=f"{s.name}_{f.name}_binning", deposition_functor=f, axes=POSITION_AXES, species=s)
        for s in species
        for f in functors
    ]


def generate_diagnostics(species, functors):
    return (
        generate_particle_dumps(species)
        + generate_native_field_dumps()
        + generate_derived_field_dumps(species, functors)
        + generate_derived_field_dumps_as_binnings(species, functors)
    )


RUN_DIR = ""


def setup_sim():
    sim = basic_simulation()
    for species in SPECIES:
        sim.add_species(species, LAYOUT)
    sim.diagnostics = [Checkpoint(TimeStepSpec[:])] + generate_diagnostics(SPECIES, FUNCTORS)
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

    def test_native_field_dump(self):
        for diag in self.sim.diagnostics:
            if isinstance(diag, NativeFieldDump):
                np.testing.assert_allclose(
                    load_diagnostic_result(diag, self.result_path),
                    read_fields(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5")[
                        diag.fieldname
                    ],
                )

    def test_compare_derived_fields_and_binning(self):
        for dump, binning in zip(
            generate_derived_field_dumps(SPECIES, FUNCTORS), generate_derived_field_dumps_as_binnings(SPECIES, FUNCTORS)
        ):
            np.testing.assert_allclose(
                load_diagnostic_result(dump, self.result_path), load_diagnostic_result(binning, self.result_path)
            )


if __name__ == "__main__":
    main()
