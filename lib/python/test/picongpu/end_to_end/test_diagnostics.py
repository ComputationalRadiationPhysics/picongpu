"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
from functools import partial
from hashlib import sha256 as compute_hash
from itertools import chain
from pathlib import Path
from unittest import TestCase, main

import numpy as np
import pandas as pd
from picongpu import rc_params
from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    FilteredSpecies,
    OnePosition,
    ParticleFilter,
    ParticleFunctor,
    Simulation,
    Species,
)
from picongpu.picmi.diagnostics import (
    Binning,
    BinningAxis,
    BinSpec,
    Checkpoint,
    DerivedFieldDump,
    NativeFieldDump,
    OpenPMDConfig,
    ParticleDump,
    TimeStepSpec,
)
from picongpu.picmi.diagnostics.backend_config import RangeSpec
from picongpu.picmi.layout import OnePositionLayout
from picongpu.picmi.particle_functor import Particle
from picongpu.picmi.particle_functor.rng_arg import RNGArg
from sympy import And, Eq, Piecewise

from .arbitrary_parameters import CELL_SIZE, NUMBER_OF_CELLS, UPPER_BOUNDARY, directory_in_home, gather_results
from .compare_particles import (
    apply_range,
    load_diagnostic_result,
    read_densities_into_mesh,
    read_fields,
    read_particles,
    sort_particles,
)
from .distributions import Gaussian, SphereFlanks

logging.basicConfig(level=logging.INFO)

LAYOUT = OnePositionLayout(n_macroparticles_per_cell=2)
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


@ParticleFunctor
def macroparticle_counter(_) -> int:
    return 1


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
    macroparticle_counter,
    # Somehow off by some factor:
    ParticleFunctor(
        name="mid_current_density_x",
        functor=lambda p: (
            p.get("charge") / np.prod(CELL_SIZE) * p.get("momentum")[0] / (p.get("gamma") * p.get("mass"))
        ),
    ),
    ParticleFunctor(name="momentum_y", functor=lambda p: p.get("momentum")[1]),
    # Duplicated just to test what happens:
    # ParticleFunctor(name="momentum_y", functor=lambda p: p.get("momentum")[1]),
    ParticleFunctor(name="momentum_density_z", functor=lambda p: p.get("momentum")[2] / np.prod(CELL_SIZE)),
    ParticleFunctor(name="weighted_velocity_x", functor=lambda p: p.get("velocity")[0] * p.get("weighting")),
]


def in_range_expression(p, r):
    if r.data is None:
        return True
    if isinstance(r.data, int):
        return Eq(p, r.data)
    return And(p >= r.data[0], p < r.data[1])


def range_filter(particle, range):
    # Technically speaking, the origin should be "global"
    # to match what the range argument of the openPMD plugin does.
    # But that's not implemented for filters.
    # "total" is identical because we don't have moving window active.
    pos = particle.get("position", unit="cell", origin="total")
    return And(*(in_range_expression(p, r) for p, r in zip(pos, range.data)))


def generate_range_restricted_particle_dumps(species):
    options = OpenPMDConfig(
        file="other_name", ext=".h5", infix="", data_preparation_strategy="doubleBuffer", range=[17, (25, 40), None]
    )
    return [
        ParticleDump(species=species[0], options=options),
        ParticleDump(
            species=FilteredSpecies(
                species=species[0],
                functor=ParticleFilter(name="rangeFilter", functor=partial(range_filter, range=options.range)),
            ),
            options=OpenPMDConfig(file="filtered"),
        ),
    ]


def generate_range_filtered_densities(species, functors):
    filtered_species = FilteredSpecies(
        species=species[0],
        functor=ParticleFilter(name="rangeFilter", functor=partial(range_filter, range=RangeSpec(17, (25, 40), None))),
    )

    density = next(f for f in functors if f.name == "density")
    return [
        DerivedFieldDump(species=filtered_species, functor=density, options={"file": "filtered_density"}),
        Binning(name="filtered_binning", deposition_functor=density, axes=POSITION_AXES, species=filtered_species),
    ]


def generate_particle_dumps(species):
    return [ParticleDump(species=s) for s in species]


def generate_native_field_dumps():
    return [NativeFieldDump(fieldname=fieldname) for fieldname in ["E", "B"]]


def generate_derived_field_dumps(species, functors):
    return [DerivedFieldDump(species=s, functor=f) for s in species for f in functors]


def _compute_threshold(distribution, rng, percent):
    count = np.prod(distribution.get("shape", 1))
    ppf = 0 if percent == 0 else (percent / 100) ** (1 / count)
    return rng.to_scipy(**distribution).ppf(ppf)


class RandomParticleFilter(ParticleFilter):
    def __init__(self, percent: int, name, distribution):
        def f(_: Particle, rng: RNGArg):
            nums = rng.get(**distribution)

            # If we've requested a specific shape of our random numbers,
            # we do a quick check that we actually got what we asked for:
            assert np.shape(nums) == distribution.get("shape", tuple())
            # But it's just for checking that it works.
            # We don't actually use it here:
            nums = np.reshape(nums, -1)

            # This threshold results in a probablity of `percent`
            # that a particle passes the full filter:
            threshold = _compute_threshold(distribution, rng, percent)
            return And(*(num < threshold for num in nums))

        super().__init__(name=f"random_filtered_{name}_{percent}", functor=f)
        self.distribution = distribution
        self.percent = percent


def _name_of(distribution):
    # CAUTION: stackoverflow suggests that this might be the better way
    # (mostly because of the efficiency of hashing vs. sorting):
    #
    #   unique_name = compute_hash(str(frozenset(distribution.items())).encode()).hexdigest()
    #
    # But it turns out that the above does not yield reproducible results between different runs.
    # Probing the following a few times on my local machine
    # found it to be sufficiently reproducible to work with conveniently:
    unique_name = compute_hash(str(tuple(sorted(distribution.items()))).encode()).hexdigest()
    return f"{distribution['dist']}_{unique_name}"


def generate_random_field_dumps(species, distribution):
    name = _name_of(distribution)
    return [
        DerivedFieldDump(
            species=FilteredSpecies(
                species=s, functor=RandomParticleFilter(percent, name=name, distribution=distribution)
            ),
            functor=macroparticle_counter,
            options={"file": f"random_{name}"},
        )
        for percent in range(0, 100, 10)
        for s in species
    ]


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


RANDOM_DISTRIBUTION = [
    {"dist": "uniform"},
    {"dist": "normal"},
    {"dist": "uniform", "range": (-42.1, 0.7), "return_type": "float_X"},
    {"dist": "uniform", "range": (100, 200), "return_type": int},
    {"dist": "normal", "std": 10.0, "mean": 7.0},
    {"dist": "normal", "shape": (2, 3)},
]


def generate_diagnostics(species, functors):
    return (
        list(chain(*(generate_random_field_dumps(species, distribution) for distribution in RANDOM_DISTRIBUTION)))
        + generate_particle_dumps(species)
        + generate_range_restricted_particle_dumps(species)
        + generate_native_field_dumps()
        + generate_derived_field_dumps(species, functors)
        + generate_range_filtered_densities(species, functors)
        + generate_derived_field_dumps_as_binnings(species, functors)
    )


# A quick switch to work with existing results.
# If you've already compiled and run these tests once
# and you're only working on your assertions,
# you can re-use your previously generated simulation results.
# (This saves a huge amount of (compilation) time!)
# Just put the `run_dir` printed in your original run here.
# DO NOT CHECK IN A NON-EMPTY STRING HERE!
RUN_DIR = ""


def setup_sim():
    sim = basic_simulation()
    for species in SPECIES:
        sim.add_species(species, LAYOUT)
    sim.diagnostics = [Checkpoint(TimeStepSpec[:])] + generate_diagnostics(SPECIES, FUNCTORS)
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


class TestDiagnostics(TestCase):
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

    def test_particle_dump(self):
        for diag in generate_particle_dumps(SPECIES):
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

    def test_compare_filtered_and_range(self):
        range_arg, filtered = generate_range_restricted_particle_dumps(SPECIES)
        np.testing.assert_allclose(
            sort_particles(load_diagnostic_result(range_arg, self.result_path)),
            sort_particles(load_diagnostic_result(filtered, self.result_path)),
        )

    def test_compare_filtered_particles_and_derived_density(self):
        _, particle_diagnostic = generate_range_restricted_particle_dumps(SPECIES)
        name = particle_diagnostic.species.species.name + "_" + particle_diagnostic.species.functor.name
        for density in generate_range_filtered_densities(SPECIES, FUNCTORS):
            np.testing.assert_allclose(
                read_densities_into_mesh(
                    load_diagnostic_result(particle_diagnostic, self.result_path), NUMBER_OF_CELLS, CELL_SIZE
                )[*name.split("_", maxsplit=1)].swapaxes(0, -1),
                load_diagnostic_result(density, self.result_path),
            )

    def test_total_number_of_random_particles(self):
        full_number_of_particles = (
            read_particles(self.result_path / "simOutput" / "checkpoints" / "checkpoint_000000.bp5")["weighting"]
            .groupby(["setup", "impl"])
            .count()
        )
        found = (
            np.round(
                pd.DataFrame.from_records(
                    [
                        (
                            _name_of(distribution),
                            *dump.species.species_name.split("_", maxsplit=1),
                            dump.species.functor.percent,
                            np.sum(load_diagnostic_result(dump, self.result_path)),
                        )
                        for distribution in RANDOM_DISTRIBUTION
                        for dump in generate_random_field_dumps(SPECIES, distribution=distribution)
                    ],
                    columns=["distribution", "setup", "impl", "expected_percent", "particle_number"],
                    index=["distribution", "setup", "impl", "expected_percent"],
                )
                .unstack(["distribution", "expected_percent"])
                .T
                / full_number_of_particles
                * 100
            )
            .T.stack(["distribution", "expected_percent"], future_stack=True)
            .astype(int)
            .reset_index("expected_percent", drop=False)
            .rename({"particle_number": "actual_percent"}, axis=1)
        )
        pd.testing.assert_series_equal(found["actual_percent"], found["expected_percent"], check_names=False)
