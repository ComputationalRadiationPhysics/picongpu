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
import openpmd_api as opmd
from openpmd_api.openpmd_api_cxx import ErrorWrongAPIUsage
from picongpu.picmi import (
    Cartesian3DGrid,
    ElectromagneticSolver,
    GriddedLayout,
    ParticleFunctor,
    Simulation,
    Species,
)
from picongpu.picmi.diagnostics import DerivedFieldDump
from picongpu.picmi.particle_functor.particle_functor import MacroParticle, PhysicalParticle
from picongpu.picmi.particle_functor.unit_dimension import M

from .arbitrary_parameters import NUMBER_OF_CELLS, UPPER_BOUNDARY, gather_results
from .distributions import Gaussian

logging.basicConfig(level=logging.INFO)

LAYOUT = GriddedLayout(n_macroparticles_per_cell=[1, 1, 1])
SPECIES = Species(
    name="Gaussian_predefined",
    particle_type="electron",
    initial_distribution=Gaussian().distributions["predefined"],
    particle_shape="other:counter",
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


# Acts on macroparticles directly (notice the argument's type annotation).
@ParticleFunctor(unit_dimension=M)
def macroparticle_mass(particle: MacroParticle) -> float:
    return particle.get("mass")


# Implementation-wise this still acts on macroparticles (everything does)
# but we scale the result manually such that it effectively acts on physical particles.
# (Scaling only happens because the argument's type annotation
# identifies this as acting on physical particles.)
@ParticleFunctor(unit_dimension=M, scales_with_weighting=1)
def manually_scaled_mass(particle: PhysicalParticle) -> float:
    return particle.get("mass")


# Implementation-wise this still acts on macroparticles
# but the annotation says that semantically this acts on physical particles.
# As we don't have any idea how to scale this with the scaling,
# we will symbolically replace all the symbols
# with their corresponding weighted version.
# This might lead to additional arithmetics in the generated code.
@ParticleFunctor(unit_dimension=M)
def physical_mass(particle: PhysicalParticle) -> float:
    return particle.get("mass")


# Result type explicitly given.
@ParticleFunctor(return_type=int)
def macroparticle_counter(_: MacroParticle):
    return 1


# Result type is inferred from the type annotation. The value is the number of
# physical particles represented by one macroparticle (i.e. the weighting,
# ~1e23 here), so it is a real quantity, not an int: a C++ `int` return would
# overflow. `scales_with_weighting=-1` turns `1` into `weighting`.
@ParticleFunctor(scales_with_weighting=-1)
def physical_counter(_: PhysicalParticle) -> float:
    return 1


@ParticleFunctor
def macro_mass_over_charge(particle: MacroParticle) -> float:
    return particle.get("mass") / particle.get("charge")


@ParticleFunctor
def physical_mass_over_charge(particle: PhysicalParticle) -> float:
    return particle.get("mass") / particle.get("charge")


DIAGNOSTICS = {
    functor.name: DerivedFieldDump(species=SPECIES, functor=functor)
    for functor in [
        macroparticle_mass,
        physical_mass,
        manually_scaled_mass,
        macroparticle_counter,
        physical_counter,
        macro_mass_over_charge,
        physical_mass_over_charge,
    ]
}


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
    sim.add_species(SPECIES, LAYOUT)
    sim.diagnostics = list(DIAGNOSTICS.values())
    if RUN_DIR:
        sim.picongpu_get_runner().run_dir = RUN_DIR
    else:
        sim.step(0)
    return sim


SIM = None


def read_fields_si(series_name, names=("E", "B")):
    """Read derived-field meshes scaled to SI via their openPMD ``unitSI``.

    Unlike the shared ``compare_particles.read_fields`` (which returns the raw
    internal-unit values and is used to compare two internal-quantity
    diagnostics against each other), the assertions here compare single
    diagnostics against absolute SI values (e.g. the species mass), so the
    ``unitSI`` scale must be applied.
    """
    series = opmd.Series(str(series_name), opmd.Access.read_only)
    # ``load_chunk`` only schedules a read; the buffers are undefined until
    # ``series.flush()``. Collect the raw chunks first, flush, then scale, so the
    # arithmetic never touches uninitialised memory.
    raw = {}
    scales = {}
    scalar = {}
    for name in names:
        mesh = series.iterations[0].meshes[name]
        try:
            raw[name] = [mesh[c].load_chunk() for c in "xyz"]
            scales[name] = [mesh[c].unit_SI for c in "xyz"]
        except ErrorWrongAPIUsage:
            scalar[name] = True
            raw[name] = mesh.load_chunk()
            scales[name] = mesh.unit_SI
    series.flush()
    tmp = {}
    for name in names:
        if scalar.get(name):
            tmp[name] = np.asarray(raw[name], dtype=np.float64) * scales[name]
        else:
            tmp[name] = [np.asarray(chunk, dtype=np.float64) * scale for chunk, scale in zip(raw[name], scales[name])]
    series.close()
    return {key: np.array(value) for key, value in tmp.items()}


class TestParticleFunctorSemantics(TestCase):
    _result_path = None

    def setUp(self):
        global SIM
        if SIM is None:
            SIM = setup_sim()
            self.sim = SIM
            gather_results(self.result_path)
        self.sim = SIM
        fieldnames_to_names = {diag.fieldname: name for name, diag in DIAGNOSTICS.items()}
        self.fields = {
            fieldnames_to_names[key]: value
            for key, value in read_fields_si(
                self.result_path / "simOutput" / "openPMD" / "simData_000000.bp5", names=fieldnames_to_names.keys()
            ).items()
        }

    @property
    def result_path(self):
        if self._result_path is None:
            self._result_path = Path(self.sim.picongpu_get_runner().run_dir)
        return self._result_path

    def test_physical_mass_equals_species_mass(self):
        np.testing.assert_allclose(
            self.fields["physical_mass"][self.fields["physical_mass"] != 0], SPECIES.picongpu_get_mass_si()
        )

    def test_manually_scaled_mass_equals_physical_mass(self):
        np.testing.assert_allclose(self.fields["manually_scaled_mass"], self.fields["physical_mass"])

    def test_mass_times_physical_particle_count_equals_macroparticle_mass(self):
        np.testing.assert_allclose(
            self.fields["physical_counter"] * self.fields["physical_mass"], self.fields["macroparticle_mass"]
        )

    def test_one_macroparticle_per_cell_everywhere(self):
        np.testing.assert_allclose(self.fields["macroparticle_counter"][self.fields["macroparticle_counter"] != 0], 1)

    def test_mass_over_charge_does_not_scale(self):
        np.testing.assert_allclose(self.fields["macro_mass_over_charge"], self.fields["physical_mass_over_charge"])


if __name__ == "__main__":
    main()
