"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BeforeValidator
from picmistandard import PICMI_BinomialSmoother, PICMI_ElectromagneticSolver

from picongpu.pypicongpu import util
from picongpu.pypicongpu.field_solver import AnySolver, LeheSolver, YeeSolver


def _single_pass_n_pass(value: Sequence[int] | None) -> Sequence[int] | None:
    """
    BeforeValidator for `BinomialSmoother.n_pass`.

    PIConGPU's C++ current interpolation applies exactly one fixed binomial
    pass (`numPasses=1` is hard-coded in
    include/picongpu/fields/currentInterpolation/Binomial.hpp), so the only
    accepted inputs are the single-pass spellings carrying that semantics: the
    standard default `None`, scalar `1` (coerced to `[1]`, the standard's
    per-axis form), and any all-ones vector (`[1]`, `[1,1]`, `[1,1,1]`, ...).
    Anything else claims a different number of passes and is rejected.
    """

    if value is None or (isinstance(value, (list, tuple)) and len(value) > 0 and all(n == 1 for n in value)):
        return value
    if value == 1:
        return [1]
    raise util.UnsupportedFeatureError(
        "more than one binomial smoothing pass (PIConGPU's C++ current interpolation "
        "applies exactly one fixed pass, hard-coded as `numPasses=1` in "
        "include/picongpu/fields/currentInterpolation/Binomial.hpp)",
        value,
    )


class BinomialSmoother(PICMI_BinomialSmoother):
    """
    PICMI Binomial Smoother

    PIConGPU's binomial current deposition is a fixed, single-pass filter: the
    C++ side hard-codes `numPasses=1` (see
    include/picongpu/fields/currentInterpolation/Binomial.hpp) and never reads
    `n_pass`. All standard parameters except `n_pass` are rejected, and
    `n_pass` may only carry that single-pass semantics: `None`, scalar `1`, or
    an all-ones vector (`[1]`, `[1,1]`, `[1,1,1]`, ...). Any value claiming
    more than one pass is rejected at construction.
    """

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
