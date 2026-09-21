"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

import math
from collections.abc import Sequence
from typing import Annotated, Any, Literal, Self

from picmistandard import PICMI_BinomialSmoother, PICMI_ElectromagneticSolver
from pydantic import BeforeValidator, ConfigDict, PrivateAttr, model_validator

from picongpu.pypicongpu import util
from picongpu.pypicongpu.field_solver import (
    AnySolver,
    ArbitraryOrderFDTDSolver,
    CKCSolver,
    LeheSolver,
    NoneSolver,
    YeeSolver,
)

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


def _ao_fDTD_weight_sum(neighbors: int) -> float:
    """
    Alternating finite-difference weight sum of the C++
    ``maxwellSolver::ArbitraryOrderFDTD`` CFL checker
    (``include/picongpu/fields/MaxwellSolver/ArbitraryOrderFDTD/ArbitraryOrderFDTD.hpp``),
    computed from ``AOFDTDWeights`` (``Weights.hpp``).

    The AO CFL limit is the Yee cell-size term divided by this factor
    (``maxC_DT = 1 / (F * sqrt(sum 1/dx^2))``), so a factor above 1 makes the
    limit *tighter* (smaller ``c * dt``) than Yee. It is ``1.0`` for one
    neighbor (plain Yee), ``7/6`` for two (order 4, so the AO limit is
    ``6/7 ~ 0.857x`` the Yee value) and ``~1.28631`` for four (order 8,
    ``~0.7774x`` the Yee value).
    """
    weights = [0.0] * neighbors
    weights[0] = (
        4.0 * neighbors * (math.factorial(2 * neighbors) / (2 ** (2 * neighbors) * math.factorial(neighbors) ** 2)) ** 2
    )
    for k in range(1, neighbors):
        weights[k] = -((k - 0.5) ** 2 * (neighbors - k) / (neighbors + k) / (k + 0.5) ** 2) * weights[k - 1]
    return sum(w if i % 2 == 0 else -w for i, w in enumerate(weights))


def _normalize_stencil_order(stencil_order: Sequence[int]) -> int:
    """
    Reduce the per-axis ``stencil_order`` to a single value.

    PIConGPU's ``ArbitraryOrderFDTD`` uses one neighbor count along every
    direction (``T_neighbors``, order = ``2 * T_neighbors``), so the per-axis
    order must be all-axes-equal. Returns that common order (>= 2, even), which
    the caller maps to ``neighbors = order // 2``.
    """
    if not stencil_order:
        util._handle_unsupported("an empty stencil_order (give a per-axis vector such as [4, 4, 4])", stencil_order)
    orders = list(stencil_order)
    if any(type(o) is not int for o in orders):
        util._handle_unsupported(
            "a stencil_order that is not a vector of ints (give ints such as [4, 4, 4])", stencil_order
        )
    if any(o < 2 for o in orders):
        util._handle_unsupported("a stencil_order with an order below 2 (each axis order must be >= 2)", stencil_order)
    if len(set(orders)) > 1:
        util._handle_unsupported(
            "a non-uniform stencil_order (PIConGPU's arbitrary-order FDTD uses the same order along every axis)",
            stencil_order,
        )
    if orders[0] % 2 != 0:
        util._handle_unsupported(
            "an odd stencil_order (the arbitrary-order FDTD order is 2 * neighbors, hence even)",
            stencil_order,
        )
    return orders[0]


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

    Supported `method` values:

    - ``"Yee"``: the standard second-order Yee solver (fixed-order, no stencil).
    - ``"Lehe"``: the Cherenkov-free Lehe solver (fixed-order, no stencil).
    - ``"CKC"``: the extended Cole-Karkkainen-Cowan solver (fixed-order, no stencil).
    - ``"other:ArbitraryOrderFDTD"``: the arbitrary-order FDTD solver; the per-axis
      ``stencil_order`` selects the order (see ``stencil_order`` below).
    - ``"other:None"``: disables the vacuum update of E and B (no CFL limit).

    PIConGPU implements the arbitrary-order FDTD with a single neighbor count
    shared by all axes (``order = 2 * neighbors``), so ``stencil_order`` is only
    meaningful for ``"other:ArbitraryOrderFDTD"`` and must be all-axes-equal; any
    ``stencil_order`` on a fixed-order solver, and any other solver options that
    PIConGPU does not implement, are rejected at construction time.
    """

    field_smoother: Annotated[PICMI_BinomialSmoother | None, util.rejects_unsupported("field smoothers")] = None
    method: Literal["Yee", "Lehe", "CKC", "other:ArbitraryOrderFDTD", "other:None"]
    stencil_order: Sequence[int] | None = None
    subcycling: Annotated[int | None, util.rejects_unsupported("subcycling")] = None
    galilean_velocity: Annotated[Sequence[float] | None, util.rejects_unsupported("galilean velocity")] = None
    divE_cleaning: Annotated[bool | None, util.rejects_unsupported("divE cleaning")] = None
    divB_cleaning: Annotated[bool | None, util.rejects_unsupported("divB cleaning")] = None
    pml_divE_cleaning: Annotated[bool | None, util.rejects_unsupported("pml divE cleaning")] = None
    pml_divB_cleaning: Annotated[bool | None, util.rejects_unsupported("pml divB cleaning")] = None

    _stencil_neighbors: int | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_stencil_order(self) -> Self:
        if self.method == "other:ArbitraryOrderFDTD":
            if self.stencil_order is None:
                util._handle_unsupported(
                    "an 'other:ArbitraryOrderFDTD' solver without a stencil_order "
                    "(give a uniform per-axis order such as [4, 4, 4])",
                    self.stencil_order,
                )
            self._stencil_neighbors = _normalize_stencil_order(self.stencil_order) // 2
        else:
            if self.stencil_order is not None:
                util._handle_unsupported(
                    "a stencil_order with a method that uses a fixed-order stencil "
                    f"(the {self.method} solver takes no stencil order)",
                    self.stencil_order,
                )
            self._stencil_neighbors = None
        return self

    def _cfl_max_cdt(self, *cell_size: float) -> float | None:
        """
        The CFL stability limit as a maximum of ``c * delta_t`` for this solver,
        mirroring the C++ ``maxwellSolver::CFLChecker`` specializations.

        ``*cell_size`` are the cell lengths along the *spatial* dimensions the
        grid resolves (three for 3D3V, two for 2D3V): passing only the spatial
        components is what makes this dimension-aware, so the same limit works
        for 2D and 3D grids.

        - ``Yee``/``Lehe``: ``1 / sqrt(sum 1/dx_i^2)`` over the spatial dims.
        - ``other:ArbitraryOrderFDTD``: the Yee term divided by the alternating
          finite-difference weight sum (``_ao_fDTD_weight_sum``), e.g. ``7/6`` for
          order 4 (AO limit ``6/7 ~ 0.857x`` the Yee value, i.e. tighter).
        - ``CKC``: the minimum spatial cell size.
        - ``other:None``: ``None`` (the solver has no CFL limit; skips the CFL gate).
        """
        if self.method == "other:None":
            return None
        inv_cell_sum = sum(1 / d**2 for d in cell_size)
        if self.method in ("Yee", "Lehe"):
            return 1 / math.sqrt(inv_cell_sum)
        if self.method == "CKC":
            return min(cell_size)
        # other:ArbitraryOrderFDTD
        return 1 / (_ao_fDTD_weight_sum(self._stencil_neighbors) * math.sqrt(inv_cell_sum))

    def get_as_pypicongpu(self) -> AnySolver:
        match self.method:
            case "Yee":
                return YeeSolver()
            case "Lehe":
                return LeheSolver()
            case "CKC":
                return CKCSolver()
            case "other:ArbitraryOrderFDTD":
                return ArbitraryOrderFDTDSolver(neighbors=self._stencil_neighbors)
            case "other:None":
                return NoneSolver()
