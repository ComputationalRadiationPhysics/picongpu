"""Tests for PICMI native and combined derived-field diagnostics."""

import pytest
from pydantic import ValidationError

from picongpu.picmi import Species
from picongpu.picmi.diagnostics import AverageDerivedFieldDump, NativeDerivedFieldDump


@pytest.fixture
def species():
    return Species(name="electrons", particle_type="electron")


@pytest.mark.parametrize(
    ("field", "expected_name"),
    [
        ("Density", "density"),
        ("BoundElectronDensity", "boundElectronDensity"),
        ("ChargeDensity", "chargeDensity"),
        ("Counter", "particleCounter"),
        ("Energy", "particleEnergy"),
        ("EnergyDensity", "energyDensity"),
        ("LarmorPower", "larmorPower"),
        ("MacroCounter", "macroParticleCounter"),
        ("RelativisticDensity", "relativisticDensity"),
        ("ScreeningInvSquared", "invSquaredScreenLength"),
    ],
)
def test_native_scalar_and_combined_fields(species, field, expected_name):
    diagnostic = NativeDerivedFieldDump(species=species, field=field)

    assert diagnostic.fieldname == f"electrons_all_{expected_name}"
    namespace = "combinedAttributes" if field in ("RelativisticDensity", "ScreeningInvSquared") else "derivedAttributes"
    assert diagnostic.get_builtin_solver()[0] == f"deriveField::{namespace}::{field}"


@pytest.mark.parametrize(
    ("field", "native_name"),
    [
        ("MidCurrentDensityComponent", "midCurrentDensity"),
        ("Momentum", "particleMomentum"),
        ("MomentumDensity", "momentumDensity"),
        ("WeightedVelocity", "weightedVelocity"),
    ],
)
def test_native_directional_fields(species, field, native_name):
    diagnostic = NativeDerivedFieldDump(species=species, field=field, direction="y")

    assert diagnostic.fieldname == f"electrons_all_{native_name}/y"
    assert diagnostic.get_builtin_solver()[0] == f"deriveField::derivedAttributes::{field}<1>"


def test_average_weighted_velocity(species):
    diagnostic = AverageDerivedFieldDump(species=species, field="WeightedVelocity", direction="z")

    assert diagnostic.fieldname == "electrons_all_Average_weightedVelocity/z"
    assert diagnostic.get_builtin_solver()[0] == (
        "deriveField::combinedAttributes::AverageAttribute<deriveField::derivedAttributes::WeightedVelocity<2>>"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"field": "Momentum"},
        {"field": "Density", "direction": "x"},
        {"field": "EnergyDensityCutoff"},
    ],
)
def test_invalid_native_field_configuration(species, kwargs):
    with pytest.raises(ValidationError):
        NativeDerivedFieldDump(species=species, **kwargs)


@pytest.mark.parametrize("field", ["MacroCounter", "RelativisticDensity", "ScreeningInvSquared"])
def test_non_averageable_fields_are_rejected(species, field):
    with pytest.raises(ValidationError):
        AverageDerivedFieldDump(species=species, field=field)
