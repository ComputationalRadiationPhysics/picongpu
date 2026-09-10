"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import os
import tempfile

import pytest

from picongpu import picmi
from picongpu.picmi.interaction.ionization.fieldionization import ADK, ADKVariant, BSI, BSIExtension, Keldysh
from picongpu.picmi.interaction.ionization.fieldionization.ionizationcurrent import (
    EnergyConservation,
    IonizationCurrent,
)
from picongpu.picmi.species import Species
from picongpu.pypicongpu.rendering import Renderer
from picongpu.pypicongpu.species.constant.ionizationcurrent import None_
from picongpu.pypicongpu.species.constant.ionizationmodel import ADKLinearPolarization

# the mustache block that renders the ionization current into the species definition
_CURRENT_TEMPLATE = (
    "particles::ionization::{{{ionizer_picongpu_name}}}<{{{ionization_electron_species.typename}}}"
    "{{#ionization_current}}, particles::ionization::current::{{{picongpu_name}}}{{/ionization_current}}>"
)


def get_grid(n: int = 32):
    return picmi.Cartesian3DGrid(
        number_of_cells=[n, n, n],
        lower_bound=[0, 0, 0],
        upper_bound=[n, n, n],
        # required, otherwise won't spawn
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"],
    )


def _adk(ionization_current, e, ion, variant=ADKVariant.LinearPolarization):
    return ADK(
        ADK_variant=variant,
        ionization_current=ionization_current,
        ion_species=ion,
        ionization_electron_species=e,
    )


def _bsi(ionization_current, e, ion, extensions=()):
    return BSI(
        BSI_extensions=list(extensions),
        ionization_current=ionization_current,
        ion_species=ion,
        ionization_electron_species=e,
    )


def _keldysh(ionization_current, e, ion):
    return Keldysh(
        ionization_current=ionization_current,
        ion_species=ion,
        ionization_electron_species=e,
    )


@pytest.mark.parametrize(
    "ionizer_builder",
    [
        lambda current, e, ion: _adk(current, e, ion),
        lambda current, e, ion: _adk(current, e, ion, ADKVariant.CircularPolarization),
        lambda current, e, ion: _bsi(current, e, ion, [BSIExtension.StarkShift]),
        lambda current, e, ion: _bsi(current, e, ion, [BSIExtension.EffectiveZ]),
        lambda current, e, ion: _keldysh(current, e, ion),
    ],
    ids=[
        "ADK-LinearPolarization",
        "ADK-CircularPolarization",
        "BSI-StarkShift",
        "BSI-EffectiveZ",
        "Keldysh",
    ],
)
@pytest.mark.parametrize(
    "ionization_current",
    [EnergyConservation(), None],
    ids=["EnergyConservation", "None"],
)
def test_ionization_current_rendered(ionizer_builder, ionization_current):
    e = picmi.Species(name="e", particle_type="electron")
    ion = picmi.Species(name="hydrogen", particle_type="H", charge_state=+1)
    ionizer = ionizer_builder(ionization_current, e, ion)
    rendered = _render_speciesDefinition(ionizer)
    if ionization_current is None:
        assert "particles::ionization::current::None" in rendered
        assert "particles::ionization::current::EnergyConservation" not in rendered
    else:
        assert "particles::ionization::current::EnergyConservation" in rendered


def _render_speciesDefinition(ionizer) -> str:
    sim = picmi.Simulation(
        time_step_size=17,
        max_steps=4,
        solver=picmi.ElectromagneticSolver(method="Yee", grid=get_grid()),
    )
    sim.add_species(ionizer.ionization_electron_species, None)
    sim.add_species(ionizer.ion_species, None)
    sim.picongpu_interaction = [ionizer]

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, "input")
        sim.write_input_file(output_dir)
        rendered_path = os.path.join(output_dir, "include", "picongpu", "param", "speciesDefinition.param")
        with open(rendered_path) as rendered_file:
            return rendered_file.read()


def test_none_byte_identical():
    # the picmi bridge (None -> None_()) must reproduce the exact rendering
    # context of the previous hardcoded None_() choice, i.e. byte-identical
    # rendered output for the default case
    e = Species(name="e", particle_type="electron")
    ion = Species(name="hydrogen", particle_type="H", charge_state=+1)

    via_bridge = ADK(
        ADK_variant=ADKVariant.LinearPolarization,
        ionization_current=None,
        ion_species=ion,
        ionization_electron_species=e,
    ).get_as_pypicongpu()
    via_hardcoded = ADKLinearPolarization(
        ionization_current=None_(),
        ionization_electron_species=e.get_as_pypicongpu(),
    )

    assert via_bridge.get_rendering_context() == via_hardcoded.get_rendering_context()
    assert Renderer.get_rendered_template(via_bridge.get_rendering_context(), _CURRENT_TEMPLATE) == (
        Renderer.get_rendered_template(via_hardcoded.get_rendering_context(), _CURRENT_TEMPLATE)
    )


def test_invalid_current_raises():
    class Bogus(IonizationCurrent):
        MODEL_NAME: str = "Bogus"

    e = picmi.Species(name="e", particle_type="electron")
    ion = picmi.Species(name="hydrogen", particle_type="H", charge_state=+1)
    ionizer = ADK(
        ADK_variant=ADKVariant.LinearPolarization,
        ionization_current=Bogus(),
        ion_species=ion,
        ionization_electron_species=e,
    )
    with pytest.raises(ValueError):
        _render_speciesDefinition(ionizer)
