"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import pytest
from pydantic import ValidationError

from picongpu import core, templates
from picongpu.pypicongpu.fieldabsorber import FieldAbsorber, format_cpp_float
from picongpu.pypicongpu.rendering.renderer import Renderer

_STATIC_FIELD_ABSORBER_PARAM = core.path("include") / "picongpu/param/fieldAbsorber.param"
"""the C++ default fieldAbsorber.param shipped with PIConGPU"""

_TEMPLATE = templates.path() / "include/picongpu/param/fieldAbsorber.param.mustache"


def _render(absorber: FieldAbsorber) -> str:
    context = {"grid": {"field_absorber": absorber.get_rendering_context(), "has_z": True}}
    Renderer.check_rendering_context(context)
    preprocessed = Renderer.get_context_preprocessed(context)
    return Renderer.get_rendered_template(preprocessed, _TEMPLATE.read_text())


def test_defaults():
    """default FieldAbsorber mirrors the C++ defaults"""
    absorber = FieldAbsorber()
    assert absorber.kind == "pml"
    assert absorber.thickness == ((12, 12), (12, 12), (12, 12))
    assert absorber.strength == ((1e-3, 1e-3), (1e-3, 1e-3), (1e-3, 1e-3))


def test_round_trip():
    """model_dump -> model_validate yields an equal model and an identical dump"""
    for kind in ("pml", "exponential"):
        absorber = FieldAbsorber(
            kind=kind, thickness=((13, 0), (4, 12), (32, 32)), strength=((0.5, 0.25), (1e-2, 1e-2), (1e-3, 2.5e-3))
        )
        dump = absorber.model_dump(mode="json")
        reparsed = FieldAbsorber.model_validate(dump)
        assert reparsed == absorber
        assert reparsed.model_dump(mode="json") == dump


def test_validation_negative_thickness():
    with pytest.raises(ValidationError):
        FieldAbsorber(thickness=((13, -1), (12, 12), (12, 12)))


def test_validation_negative_strength():
    with pytest.raises(ValidationError):
        FieldAbsorber(strength=((0.5, -0.1), (1e-3, 1e-3), (1e-3, 1e-3)))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_validation_non_finite_strength(bad):
    """non-finite strengths must be rejected at validation, not crash at render"""
    with pytest.raises(ValidationError):
        FieldAbsorber(strength=((bad, 1e-3), (1e-3, 1e-3), (1e-3, 1e-3)))
    with pytest.raises(ValidationError):
        FieldAbsorber(strength=((1e-3, 1e-3), (1e-3, 1e-3), (bad, 1e-3)))


def test_validation_wrong_shape():
    with pytest.raises(ValidationError):
        FieldAbsorber(thickness=((12, 12), (12, 12)))
    with pytest.raises(ValidationError):
        FieldAbsorber(strength=((1e-3, 1e-3, 1e-3), (1e-3, 1e-3, 1e-3), (1e-3, 1e-3, 1e-3)))


def test_validation_malformed_row():
    with pytest.raises(ValidationError):
        FieldAbsorber(thickness=((12, 12), "not-a-pair", (12, 12)))


def test_format_cpp_float():
    assert format_cpp_float(1e-3) == "1.0e-3"
    assert format_cpp_float(0.0) == "0.0"
    assert format_cpp_float(1.0) == "1.0"
    assert format_cpp_float(0.5) == "5.0e-1"


def test_default_render_byte_equal_to_static_file():
    """default-config render of the mustache template equals the static C++ param file byte for byte"""
    assert _render(FieldAbsorber()).encode() == _STATIC_FIELD_ABSORBER_PARAM.read_bytes()


def test_asymmetric_thickness_render():
    """asymmetric [3][2] thickness is rendered faithfully, keeping the THICKNESS symbol only for default cells"""
    absorber = FieldAbsorber(thickness=((13, 0), (4, 12), (32, 32)))
    rendered = _render(absorber)
    assert "{13, 0}, // x direction [negative, positive]" in rendered
    assert "{4, THICKNESS}, // y direction [negative, positive]" in rendered
    assert "{32, 32} // z direction [negative, positive]" in rendered


def test_exponential_strength_render():
    """exponential::STRENGTH defaults and overrides are rendered as C++ literals"""
    rendered = _render(FieldAbsorber())
    assert "{1.0e-3, 1.0e-3}, /*x direction [negative,positive]*/" in rendered

    absorber = FieldAbsorber(strength=((0.5, 0.0), (1e-2, 1e-2), (1e-3, 2.5e-3)))
    rendered = _render(absorber)
    assert "{5.0e-1, 0.0}, /*x direction [negative,positive]*/" in rendered
    assert "{1.0e-3, 2.5e-3} /*z direction [negative,positive]*/" in rendered
