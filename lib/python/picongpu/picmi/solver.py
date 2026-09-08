"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from collections.abc import Sequence
from typing import Annotated, Any, Literal

from picmistandard import PICMI_BinomialSmoother, PICMI_ElectromagneticSolver
from pydantic import BeforeValidator, ConfigDict

from picongpu.pypicongpu import util
from picongpu.pypicongpu.field_solver import AnySolver, LeheSolver, YeeSolver

_other_than_one_pass = (
    "a number of binomial smoothing passes other than one (PIConGPU's C++ current "
    "interpolation applies exactly one fixed pass, hard-coded as `numPasses=1` in "
    "include/picongpu/fields/currentInterpolation/Binomial.hpp)"
)
_not_a_single_pass_spelling = (
    "an n_pass that is neither None, scalar 1, nor an all-ones list/tuple of ints "
    "(the picmistandard types n_pass as `Sequence[int] | None`)"
)

_CANONICAL_N_PASS: tuple[int, int, int] = (1, 1, 1)


def _single_pass_n_pass(value: Any) -> Sequence[int] | None:
    """
    BeforeValidator for `BinomialSmoother.n_pass`.

    PIConGPU's C++ current interpolation applies exactly one fixed binomial
    pass (`numPasses=1` is hard-coded in
    include/picongpu/fields/currentInterpolation/Binomial.hpp), so the only
    accepted inputs are the single-pass spellings carrying that semantics: the
    standard default `None`, scalar `1`, and any all-ones vector (`[1]`,
    `[1,1]`, `[1,1,1]`, ...). Any of these is reduced to the single canonical
    form `(1, 1, 1)`.
    """

    if value is None:
        return _CANONICAL_N_PASS
    # a bespoke validator is used instead of util.rejects_unsupported, because
    # that helper may only compare a value against a single accepted `default`,
    # whereas here every standard-conformant "single pass" spelling (None, any
    # all-ones vector, scalar 1) is accepted; the rejections still go through
    # util._handle_unsupported, the single raise/warn/ignore policy point.
    if type(value) is int:
        # scalar single-pass sugar, reduced to the canonical form;
        # `type(...) is int` is strict on purpose: bool (a subclass of int),
        # float and numpy scalars are not standard-conformant `int`s
        if value == 1:
            return _CANONICAL_N_PASS
        util._handle_unsupported(_other_than_one_pass, value)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            util._handle_unsupported(
                "an empty n_pass (give None or an all-ones per-axis vector such as [1] or [1,1,1])",
                value,
            )
        if all(type(n) is int for n in value):
            if all(n == 1 for n in value):
                return _CANONICAL_N_PASS
            util._handle_unsupported(_other_than_one_pass, value)
    util._handle_unsupported(_not_a_single_pass_spelling, value)


class BinomialSmoother(PICMI_BinomialSmoother):
    """
    PICMI Binomial Smoother

    PIConGPU's binomial current deposition is a fixed, single-pass filter: the
    C++ side hard-codes `numPasses=1` (see
    include/picongpu/fields/currentInterpolation/Binomial.hpp) and never reads
    `n_pass`. All standard parameters except `n_pass` are rejected, and
    `n_pass` may only carry that single-pass semantics: `None`, scalar `1`, or
    an all-ones vector (`[1]`, `[1,1]`, `[1,1,1]`, ...). Any of these is
    reduced to the canonical form `(1, 1, 1)`, and any value claiming more
    than one pass is rejected at construction.

    Note: `n_pass` is defaulted to `None` here. The picmistandard base class
    declares it with a broken `default_factory=None`, which makes pydantic treat
    the field as *required* (`BinomialSmoother()` raised "Field required"). This
    subclass intentionally flips that: a plain `BinomialSmoother()` now
    constructs fine and means "single pass on every axis" (stored as the
    canonical `(1, 1, 1)`), and the old behaviour of requiring a (then silently
    ignored) value is deliberately dropped.
    """

    model_config = ConfigDict(validate_default=True)

    n_pass: Annotated[Sequence[int] | None, BeforeValidator(_single_pass_n_pass)] = None
    compensation: Annotated[Sequence[bool] | None, util.rejects_unsupported("binomial smoother parameters")] = None
    stride: Annotated[Sequence[int] | None, util.rejects_unsupported("binomial smoother parameters")] = None
    alpha: Annotated[Sequence[float] | None, util.rejects_unsupported("binomial smoother parameters")] = None


class ElectromagneticSolver(PICMI_ElectromagneticSolver):
    """
    PICMI Electromagnic Solver

    See PICMI spec for full documentation.

    Only the Yee and Lehe solvers are supported; solver options that PIConGPU
    does not implement are rejected at construction time.
    """

    field_smoother: Annotated[PICMI_BinomialSmoother | None, util.rejects_unsupported("field smoothers")] = None
    method: Literal["Yee", "Lehe"]
    stencil_order: Annotated[Sequence[int] | None, util.rejects_unsupported("higher order solver stencils")] = None
    subcycling: Annotated[int | None, util.rejects_unsupported("subcycling")] = None
    galilean_velocity: Annotated[Sequence[float] | None, util.rejects_unsupported("galilean velocity")] = None
    divE_cleaning: Annotated[bool | None, util.rejects_unsupported("divE cleaning")] = None
    divB_cleaning: Annotated[bool | None, util.rejects_unsupported("divB cleaning")] = None
    pml_divE_cleaning: Annotated[bool | None, util.rejects_unsupported("pml divE cleaning")] = None
    pml_divB_cleaning: Annotated[bool | None, util.rejects_unsupported("pml divB cleaning")] = None

    def get_as_pypicongpu(self) -> AnySolver:
        return YeeSolver() if self.method == "Yee" else LeheSolver()
