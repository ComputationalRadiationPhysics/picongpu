"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from collections.abc import Iterable
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, Field, PlainSerializer, computed_field

from .rendering import RenderedObject


def _validate_absorber_matrix(values, *, field: str):
    """
    Validate a per-axis x per-boundary ``[3][2]`` matrix, mirroring ``NUM_CELLS`` / ``exponential::STRENGTH``.

    Every entry must be present and greater than or equal to 0.
    """
    try:
        flattened = [values[axis][boundary] for axis in range(3) for boundary in range(2)]
    except (IndexError, TypeError) as error:
        raise ValueError(f"{field=} must have shape [3][2], but got {values=}.") from error
    if wrong := [x for x in flattened if x < 0]:
        raise ValueError(f"{field=} contains negative values {wrong=}, which is not allowed.")
    return values


def _normalise_matrix(values, field: str):
    """Accept both the nested-sequence and the serialised per-axis dict form to allow round-tripping."""
    if isinstance(values, Iterable) and values and isinstance(values[0], dict):
        if not all(isinstance(axis, dict) for axis in values):
            raise ValueError(f"{field=} must contain 'negative' and 'positive' per axis, got {values=}.")
        try:
            return tuple((axis["negative"], axis["positive"]) for axis in values)
        except KeyError as error:
            raise ValueError(f"{field=} must contain 'negative' and 'positive' per axis, got {values=}.") from error
    return tuple(tuple(axis) for axis in values)


def _serialise_matrix(values) -> list[dict[Literal["negative", "positive"], int | float]]:
    """Serialize a ``[3][2]`` matrix as per-axis (negative, positive) pairs for mustache rendering."""
    return [{"negative": axis[0], "positive": axis[1]} for axis in values]


_int_matrix = Annotated[
    tuple[tuple[int, int], tuple[int, int], tuple[int, int]],
    BeforeValidator(lambda v: _normalise_matrix(v, "thickness")),
    AfterValidator(lambda v: _validate_absorber_matrix(v, field="thickness")),
    PlainSerializer(_serialise_matrix, return_type=list),
]
_float_matrix = Annotated[
    tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    BeforeValidator(lambda v: _normalise_matrix(v, "strength")),
    AfterValidator(lambda v: _validate_absorber_matrix(v, field="strength")),
    PlainSerializer(_serialise_matrix, return_type=list),
]


def _cpp_thickness_cell(value: int, default: int) -> str:
    """Render a single ``NUM_CELLS`` entry, keeping the ``THICKNESS`` convenience symbol where faithful."""
    if value == default:
        return "THICKNESS"
    return str(int(value))


def format_cpp_float(value: float) -> str:
    """
    Format a float as a valid C++ floating point literal.

    The default exponential strength (1e-3) is rendered as "1.0e-3" so that the
    default render matches include/picongpu/param/fieldAbsorber.param byte for byte.
    """
    if value == 0.0:
        return "0.0"
    mantissa, exponent = f"{value:.15e}".split("e")
    exponent_as_int = int(exponent)
    mantissa = mantissa.rstrip("0").rstrip(".")
    if "." not in mantissa:
        mantissa += ".0"
    if exponent_as_int == 0:
        return mantissa
    return f"{mantissa}e{exponent_as_int}"


class FieldAbsorber(RenderedObject, BaseModel):
    """
    PIConGPU field absorber configuration.

    Faithfully mirrors ``include/picongpu/param/fieldAbsorber.param``:

    - ``kind`` selects the absorber kind (command-line option ``--fieldAbsorber``),
    - ``thickness_default`` is the C++ ``THICKNESS`` convenience constant,
    - ``thickness`` mirrors ``NUM_CELLS`` (thickness of the absorbing layer in cells,
      per axis x/y/z and per boundary negative/positive; 0 disables absorption there),
    - ``strength`` mirrors ``exponential::STRENGTH`` (only used for the exponential absorber).

    Emission is kind-independent: the rendered ``fieldAbsorber.param`` always contains
    every section exactly as in the C++ file, and ``kind`` only selects the command-line
    option ``--fieldAbsorber``. This mirrors C++, where all sections are always compiled
    in and the kind is a runtime choice.

    The absorbing layer lies inside the global domain near the outer borders.
    """

    kind: Literal["pml", "exponential"] = "pml"
    """absorber kind, selects the ``--fieldAbsorber`` command line value"""

    thickness_default: int = Field(default=12, ge=0)
    """C++ ``THICKNESS`` convenience constant: the default per-side thickness in cells"""

    thickness: _int_matrix = Field(
        default_factory=lambda: ((12, 12), (12, 12), (12, 12)),
    )
    """thickness of the absorbing layer in cells, ``NUM_CELLS[3][2]`` (axis x/y/z, boundary negative/positive)"""

    strength: _float_matrix = Field(
        default_factory=lambda: ((1e-3, 1e-3), (1e-3, 1e-3), (1e-3, 1e-3)),
    )
    """strength of the exponential absorber, ``exponential::STRENGTH[3][2]`` (axis x/y/z, boundary negative/positive)"""

    @computed_field(return_type=list[dict[str, str]])
    @property
    def thickness_cpp(self) -> list[dict[str, str]]:
        """``NUM_CELLS[3][2]`` rendered as C++ literals, keeping the ``THICKNESS`` symbol where values match the default."""
        return [
            {
                "axis": axis,
                "cell": "{"
                + _cpp_thickness_cell(self.thickness[i][0], self.thickness_default)
                + ", "
                + _cpp_thickness_cell(self.thickness[i][1], self.thickness_default)
                + "}",
            }
            for i, axis in enumerate("xyz")
        ]

    @computed_field(return_type=list[dict[str, str]])
    @property
    def strength_cpp(self) -> list[dict[str, str]]:
        """``exponential::STRENGTH[3][2]`` rendered as C++ floating point literals."""
        return [
            {
                "axis": axis,
                "cell": "{"
                + format_cpp_float(self.strength[i][0])
                + ", "
                + format_cpp_float(self.strength[i][1])
                + "}",
            }
            for i, axis in enumerate("xyz")
        ]
