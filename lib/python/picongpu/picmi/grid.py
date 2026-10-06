"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch, Julian Lenz
License: GPLv3+
"""

import inspect
import warnings
from typing import Annotated, Literal, Sequence
import picmistandard
from pydantic import AfterValidator, BeforeValidator, Field, computed_field, model_validator

from ..pypicongpu import fieldabsorber, grid, util
from .copy_attributes import converts_to


def _warn_user(message: str) -> None:
    """Emit a ``UserWarning`` attributed to the first frame outside the picongpu package.

    The validation runs from several entry points (``check()`` and the
    ``get_as_pypicongpu()`` preamble) with different call depths, so a fixed
    ``stacklevel`` would either point into pydantic internals or stay inside this
    package. Walking out of the package attributes the warning to the caller.
    """
    stacklevel = 1
    frame = inspect.currentframe()
    while frame is not None and (frame.f_globals.get("__package__") or "").startswith("picongpu"):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, stacklevel=stacklevel)


def _normalise_type(kw, key, t):
    kw[key] = tuple(t(bound) for bound in kw[key])
    return kw


PICONGPU_BOUNDARY_CONDITION_BY_PICMI_ID = {
    "open": grid.BoundaryCondition.ABSORBING,
    "periodic": grid.BoundaryCondition.PERIODIC,
}


def _reject_bool_n_gpus(n_gpus):
    # bool is an int subclass, so without this explicit check pydantic's lax
    # int coercion would silently turn True/False into 1/0 GPUs.
    if isinstance(n_gpus, bool):
        raise ValueError(
            f"picongpu_n_gpus must be a positive int or a sequence of positive ints, not a bool. You gave {n_gpus!r}."
        )
    return n_gpus


def _normalise_n_gpus(n_gpus, n_dimensions: int):
    """Normalise the accepted forms of ``picongpu_n_gpus`` into an ``n_dimensions``-tuple.

    Accepted forms:
      * ``None`` -> single-GPU default ``(1, ..., 1)``
      * a bare positive int ``N`` -> parallelise in y: ``N`` in the second
        slot, 1 everywhere else (e.g. ``(1, N, 1)`` in 3D, ``(1, N)`` in 2D)
      * a 1-element sequence ``[N]`` / ``(N,)`` -> same as a bare int
      * an ``n_dimensions``-element sequence -> unchanged

    Everything else (empty, wrong-length or non-positive sequences, ...) is
    rejected. Note that pydantic's lax mode coerces whole-number floats to int
    (``4.0`` -> ``4``) before this runs, so integral floats are accepted as the
    equivalent int on purpose.
    """
    picongpu_n_gpus = n_gpus
    # a bare integer is interpreted as a single number of GPUs parallelized in y
    if n_gpus is None:
        n_gpus = tuple([1] * n_dimensions)
    elif isinstance(n_gpus, int):
        n_gpus = tuple(n_gpus if i == 1 else 1 for i in range(n_dimensions))
    else:
        n_gpus = tuple(n_gpus)

    if len(n_gpus) == 1:
        n_gpus = tuple(n_gpus[0] if i == 1 else 1 for i in range(n_dimensions))

    if len(n_gpus) != n_dimensions:
        raise ValueError(
            f"The given number of gpus could not be mapped to a {n_dimensions}-component list of integers. "
            f"You gave {picongpu_n_gpus} and we interpreted this as {n_gpus=}."
        )

    if any(map(lambda x: x <= 0, n_gpus)):
        raise ValueError(
            f"Number of gpus must be positive integer(s). "
            f"You gave {picongpu_n_gpus=} and we interpreted this as {n_gpus=}."
        )

    return n_gpus


def _check_cartesian_grid(self, dim_name):
    _check_lower_bound_is_zero(self)
    _check_boundary_conditions(self, dim_name)
    _reject_unsupported_cartesian_grid_features(self)
    _check_field_absorber(self, dim_name)
    _check_grid_distribution(self, dim_name)
    _check_super_cell_size(self, dim_name)


def _reject_unsupported_cartesian_grid_features(self):
    util.unsupported("moving window", self.moving_window_velocity)
    util.unsupported("refined regions", self.refined_regions, [])
    util.unsupported("lower bound (particles)", self.lower_bound_particles, self.lower_bound)
    util.unsupported("upper bound (particles)", self.upper_bound_particles, self.upper_bound)
    util.unsupported(
        "lower boundary conditions (particles)",
        self.lower_boundary_conditions_particles,
        self.lower_boundary_conditions,
    )
    util.unsupported(
        "upper boundary conditions (particles)",
        self.upper_boundary_conditions_particles,
        self.upper_boundary_conditions,
    )


def _as_non_negative_int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer, you gave {value!r}.")
    return value


def _as_non_negative_float(value, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be a non-negative number, you gave {value!r}.")
    value = float(value)
    if value < 0 or value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{where} must be a finite, non-negative number, you gave {value!r}.")
    return value


def _absorber_thickness_and_strength(self, n_axes: int) -> tuple[tuple, tuple]:
    """Build the full ``[3][2]`` ``NUM_CELLS`` thickness and ``exponential::STRENGTH`` matrices.

    The standard ``pml_cells`` (per-axis symmetric, both boundaries equal) and the
    PIConGPU ``picongpu_pml_cells`` extension (per-axis, per-direction
    ``[negative, positive]``) are two ways to set the depth and are mutually
    exclusive. The ``picongpu_exponential_strength`` extension sets the
    exponential ``STRENGTH`` per axis and direction. Axes beyond ``n_axes`` (the
    inert z axis in 2D) keep the ``fieldAbsorber.param`` defaults.
    """
    if self.pml_cells is not None and self.picongpu_pml_cells is not None:
        raise ValueError(
            "pml_cells and picongpu_pml_cells are two ways to configure the absorber depth; "
            "please use only one of them."
        )

    default = fieldabsorber.DEFAULT_THICKNESS
    thickness = [[default, default] for _ in range(3)]
    if self.pml_cells is not None:
        cells = list(self.pml_cells)
        if len(cells) != n_axes:
            raise ValueError(f"pml_cells must be a list of {n_axes} integers, you gave {self.pml_cells=}.")
        for axis in range(n_axes):
            cells[axis] = _as_non_negative_int(cells[axis], f"pml_cells[{axis}]")
            thickness[axis] = [cells[axis], cells[axis]]
    elif self.picongpu_pml_cells is not None:
        pairs = list(self.picongpu_pml_cells)
        if len(pairs) != n_axes:
            raise ValueError(
                f"picongpu_pml_cells must be a list of {n_axes} [negative, positive] pairs, "
                f"you gave {self.picongpu_pml_cells=}."
            )
        for axis in range(n_axes):
            pair = list(pairs[axis])
            if len(pair) != 2:
                raise ValueError(
                    f"picongpu_pml_cells[{axis}] must be a [negative, positive] pair, you gave {pairs[axis]!r}."
                )
            thickness[axis] = [
                _as_non_negative_int(pair[0], f"picongpu_pml_cells[{axis}][0] (negative)"),
                _as_non_negative_int(pair[1], f"picongpu_pml_cells[{axis}][1] (positive)"),
            ]

    default_strength = fieldabsorber.DEFAULT_STRENGTH
    strength = [[default_strength, default_strength] for _ in range(3)]
    if self.picongpu_exponential_strength is not None:
        pairs = list(self.picongpu_exponential_strength)
        if len(pairs) != n_axes:
            raise ValueError(
                f"picongpu_exponential_strength must be a list of {n_axes} [negative, positive] pairs, "
                f"you gave {self.picongpu_exponential_strength=}."
            )
        for axis in range(n_axes):
            pair = list(pairs[axis])
            if len(pair) != 2:
                raise ValueError(
                    f"picongpu_exponential_strength[{axis}] must be a [negative, positive] pair, "
                    f"you gave {pairs[axis]!r}."
                )
            strength[axis] = [
                _as_non_negative_float(pair[0], f"picongpu_exponential_strength[{axis}][0] (negative)"),
                _as_non_negative_float(pair[1], f"picongpu_exponential_strength[{axis}][1] (positive)"),
            ]

    return tuple(tuple(axis) for axis in thickness), tuple(tuple(axis) for axis in strength)


def _build_field_absorber(self, n_axes: int):
    """Translate the grid absorber knobs into a pypicongpu :class:`FieldAbsorber`."""
    thickness, strength = _absorber_thickness_and_strength(self, n_axes)
    return fieldabsorber.FieldAbsorber(
        kind=self.picongpu_absorber_kind,
        thickness=thickness,
        strength=strength,
    )


def _check_field_absorber(self, dim_name):
    """Validate the absorber knobs and warn about no-op configurations on periodic axes."""
    n_axes = len(dim_name)
    thickness, _ = _absorber_thickness_and_strength(self, n_axes)
    depth_configured = self.pml_cells is not None or self.picongpu_pml_cells is not None
    absorber_configured = (
        depth_configured
        or self.picongpu_exponential_strength is not None
        or "picongpu_absorber_kind" in self.model_fields_set
    )
    if not absorber_configured:
        return

    # A hard domain-fit error only for an *explicitly configured depth*. The
    # unconfigured default (12 cells per side, as in the static C++
    # fieldAbsorber.param) is subject to the C++ DomainAdjuster, which rounds the
    # local domain up at runtime; rejecting such grids here would break setups
    # that never asked for a custom absorber (and, as in C++, still run). Picking
    # only a profile/kind or a strength is not a depth choice and must not be
    # rejected for a depth the user never set.
    if depth_configured:
        grid.check_absorber_fits(
            self.number_of_cells,
            self.picongpu_n_gpus,
            self.picongpu_grid_dist,
            tuple(PICONGPU_BOUNDARY_CONDITION_BY_PICMI_ID[c] for c in self.lower_boundary_conditions),
            _build_field_absorber(self, n_axes),
        )

    periodic_axes = [axis for axis in range(n_axes) if self.lower_boundary_conditions[axis] == "periodic"]
    if len(periodic_axes) == n_axes:
        _warn_user(
            "All boundaries are periodic; PIConGPU forces the field absorber kind to None, "
            "so the configured absorber has no effect."
        )
        return
    # Only an explicitly configured depth can be a no-op on a periodic axis; the
    # default depth is not a configuration and must not warn.
    if depth_configured:
        for axis in periodic_axes:
            if any(thickness[axis]):
                _warn_user(
                    f"The field absorber thickness on the periodic {dim_name[axis]} axis is ignored "
                    "(PIConGPU applies no absorber there)."
                )


def _check_lower_bound_is_zero(self):
    if any(bound != 0.0 for bound in self.lower_bound):
        raise ValueError(f"A lower bound different from 0 is not supported in PIConGPU. You gave {self.lower_bound}.")


def _check_boundary_conditions(self, dim_name):
    if self.lower_boundary_conditions != self.upper_boundary_conditions:
        raise ValueError(
            "upper and lower boundary conditions must be equal (can only be chosen by axis, not by direction)"
        )
    for i, name in enumerate(dim_name):
        if self.lower_boundary_conditions[i] not in PICONGPU_BOUNDARY_CONDITION_BY_PICMI_ID:
            raise ValueError(f"{name}: boundary condition not supported")


def _check_grid_distribution(self, dim_name):
    if self.picongpu_grid_dist is None:
        return
    for i, name in enumerate(dim_name):
        if not all(n >= 1 for n in self.picongpu_grid_dist[i]):
            raise ValueError("All values in grid distribution must be greater than 0.")
        if sum(self.picongpu_grid_dist[i]) != self.number_of_cells[i]:
            raise ValueError(f"sum of grid distribution in {name} dimension must match number of cells")
        if len(self.picongpu_grid_dist[i]) != self.picongpu_n_gpus[i]:
            raise ValueError(f"number of grid distributions in {name} dimension must match number of gpus")


def _check_super_cell_size(self, dim_name):
    for i, name in enumerate(dim_name):
        if self.picongpu_super_cell_size[i] < 1:
            raise ValueError("super cell size must be a positive integer")

    if self.guard_cells is not None:
        for i, name in enumerate(dim_name):
            guard_cells = self.guard_cells[i]
            super_cell = self.picongpu_super_cell_size[i]
            if guard_cells < 0:
                raise ValueError(
                    f"guard cells in {name} dimension must be a non-negative integer. You gave {guard_cells}."
                )
            if guard_cells % super_cell != 0:
                raise ValueError(
                    f"guard cells in {name} dimension must be an exact multiple of the super cell size "
                    f"({super_cell} in {name}), but you gave {guard_cells}."
                )
    cells = list(self.number_of_cells)
    for dim, name in enumerate(dim_name):
        if self.picongpu_grid_dist is None:
            if ((cells[dim] // self.picongpu_n_gpus[dim]) // self.picongpu_super_cell_size[dim]) * self.picongpu_n_gpus[
                dim
            ] * self.picongpu_super_cell_size[dim] != cells[dim]:
                raise ValueError(
                    "GPU- and/or super-cell-distribution in {} dimension does not match grid size".format(name)
                )
        else:
            # any returns true if there is at least one non zero (True) element
            if any([x % self.picongpu_super_cell_size[dim] for x in self.picongpu_grid_dist[dim]]):
                raise ValueError(f"grid distribution in {name} dimension must be multiple of super cell size")


@converts_to(
    grid.Grid3D,
    preamble=lambda self: _check_cartesian_grid(self, ["x", "y", "z"]),
    conversions={
        "boundary_condition": lambda self: tuple(
            PICONGPU_BOUNDARY_CONDITION_BY_PICMI_ID[x] for x in self.lower_boundary_conditions
        ),
        "cell_cnt": "number_of_cells",
        "field_absorber": lambda self: _build_field_absorber(self, 3),
        "guard_size": lambda self: (
            None
            if self.guard_cells is None
            else tuple(c // s for c, s in zip(self.guard_cells, self.picongpu_super_cell_size))
        ),
    },
    remove_prefix="picongpu_",
)
class Cartesian3DGrid(picmistandard.PICMI_Cartesian3DGrid):
    # number of GPUs to distribute the grid over; whatever form is given, it
    # is normalized to a 3-tuple (see _normalise_n_gpus): a bare int N and [N]
    # both mean "parallelize over N GPUs in y", i.e. (1, N, 1)
    picongpu_n_gpus: Annotated[
        int | Sequence[int] | None,
        BeforeValidator(_reject_bool_n_gpus),
        AfterValidator(lambda x: _normalise_n_gpus(x, 3)),
    ] = Field(default=(1, 1, 1))
    picongpu_grid_dist: None | list[list[int]] = Field(default=None)
    picongpu_super_cell_size: tuple[int, int, int] = Field(default=(8, 8, 4))
    picongpu_absorber_kind: Literal["pml", "exponential"] = Field(default="pml")
    """Absorber profile/kind (``--fieldAbsorber``): ``"pml"`` or ``"exponential"``.

    PIConGPU supports exactly these two profiles (the C++ runtime option
    ``--fieldAbsorber``); both are always compiled in. On an all-periodic grid the
    C++ core overrides the choice and disables the absorber entirely.
    """
    picongpu_pml_cells: None | list[list[int]] = Field(default=None)
    """Per-axis, per-direction absorber depth ``[[negative, positive], ...]`` in cells.

    PIConGPU extension (the standard ``pml_cells`` is per-axis symmetric): exposes the
    full ``NUM_CELLS[3][2]`` matrix so the two boundaries of an axis may differ.
    Mutually exclusive with the standard ``pml_cells``.
    """
    picongpu_exponential_strength: None | list[list[float]] = Field(default=None)
    """Per-axis, per-direction ``exponential::STRENGTH`` ``[[negative, positive], ...]``.

    Only used by the ``"exponential"`` absorber; the default (``1e-3`` everywhere)
    mirrors the static C++ ``fieldAbsorber.param``.
    """

    @computed_field
    def picongpu_cell_size(self) -> tuple[int, int, int]:
        return (
            (self.upper_bound[0] - self.lower_bound[0]) / self.number_of_cells[0],
            (self.upper_bound[1] - self.lower_bound[1]) / self.number_of_cells[1],
            (self.upper_bound[2] - self.lower_bound[2]) / self.number_of_cells[2],
        )

    @model_validator(mode="after")
    def _validate(self):
        # A grid must be non-degenerate: every dimension needs at least one cell
        # and a strictly positive extent, otherwise the cell size below would be
        # undefined (ZeroDivisionError) or render an empty C++ grid.
        for dim, name in enumerate(["x", "y", "z"]):
            if self.number_of_cells[dim] < 1:
                raise ValueError(
                    f"number_of_cells[{dim}] ({name} dimension) must be a positive integer. "
                    f"You gave {self.number_of_cells[dim]}."
                )
            if self.upper_bound[dim] <= self.lower_bound[dim]:
                raise ValueError(
                    f"upper_bound in {name} dimension must be greater than lower_bound "
                    f"(got lower={self.lower_bound[dim]}, upper={self.upper_bound[dim]})."
                )
        return self

    def check(self):
        _check_cartesian_grid(self, ["x", "y", "z"])

    def to_2d(self) -> "Cartesian2DGrid":
        """Reduce this 3D grid to a 2D (2D3V) grid, dropping the z (third) component.

        Every vector field is mapped from a 3-tuple to a 2-tuple by keeping the
        (x, y) components. The z cell length is preserved as the 2D slab
        thickness via ``picongpu_cell_depth_si``. If the 3D super cell is the
        default ``(8, 8, 4)`` it maps to the 2D default ``(16, 16)``; an
        explicitly-set 3D super cell keeps its (x, y) components (detected via
        ``model_fields_set``, not by value, so an explicit ``(8, 8, 4)`` is
        preserved as ``(8, 8)``). A fresh ``Cartesian2DGrid`` is returned and the
        source grid is not modified; the result is then validated with the 2D
        ``check()`` so a 3D grid whose reduction does not satisfy the 2D
        constraints raises at the call site rather than returning an invalid grid.
        """
        number_of_cells = self.number_of_cells[:2]
        lower_bound = self.lower_bound[:2]
        upper_bound = self.upper_bound[:2]
        lower_boundary_conditions = self.lower_boundary_conditions[:2]
        upper_boundary_conditions = self.upper_boundary_conditions[:2]

        # The 3D default super cell maps to the 2D default; an explicitly-set
        # super cell (including a deliberate ``(8, 8, 4)``) keeps its (x, y).
        # ``model_fields_set`` tells explicit from default; the value cannot.
        if "picongpu_super_cell_size" in self.model_fields_set:
            super_cell_size = self.picongpu_super_cell_size[:2]
        else:
            super_cell_size = (16, 16)

        kwargs = dict(
            number_of_cells=number_of_cells,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            lower_boundary_conditions=lower_boundary_conditions,
            upper_boundary_conditions=upper_boundary_conditions,
            picongpu_super_cell_size=super_cell_size,
            picongpu_cell_depth_si=self.picongpu_cell_size[2],
        )
        if self.guard_cells is not None:
            kwargs["guard_cells"] = self.guard_cells[:2]
        if self.picongpu_n_gpus != (1, 1, 1):
            kwargs["picongpu_n_gpus"] = self.picongpu_n_gpus[:2]
        if self.picongpu_grid_dist is not None:
            kwargs["picongpu_grid_dist"] = self.picongpu_grid_dist[:2]
        # Absorber knobs are only carried over when explicitly set, so a reduced
        # grid does not look "explicitly configured" and keeps the C++ default.
        if "picongpu_absorber_kind" in self.model_fields_set:
            kwargs["picongpu_absorber_kind"] = self.picongpu_absorber_kind
        if self.pml_cells is not None:
            kwargs["pml_cells"] = self.pml_cells[:2]
        if self.picongpu_pml_cells is not None:
            kwargs["picongpu_pml_cells"] = self.picongpu_pml_cells[:2]
        if self.picongpu_exponential_strength is not None:
            kwargs["picongpu_exponential_strength"] = self.picongpu_exponential_strength[:2]
        grid_2d = Cartesian2DGrid(**kwargs)
        grid_2d.check()
        return grid_2d


@converts_to(
    grid.Grid2D,
    preamble=lambda self: _check_cartesian_grid(self, ["x", "y"]),
    conversions={
        "boundary_condition": lambda self: tuple(
            PICONGPU_BOUNDARY_CONDITION_BY_PICMI_ID[x] for x in self.lower_boundary_conditions
        ),
        "cell_cnt": "number_of_cells",
        # In 2D3V the Z cell length (CELL_DEPTH_SI) is still used to normalize
        # densities; we take the x cell size as the default wire-particle length
        # unless the user explicitly overrides it via picongpu_cell_depth_si.
        "cell_depth_si": lambda self: (
            self.picongpu_cell_depth_si
            if self.picongpu_cell_depth_si is not None
            else (self.upper_bound[0] - self.lower_bound[0]) / self.number_of_cells[0]
        ),
        "field_absorber": lambda self: _build_field_absorber(self, 2),
        "guard_size": lambda self: (
            None
            if self.guard_cells is None
            else tuple(c // s for c, s in zip(self.guard_cells, self.picongpu_super_cell_size))
        ),
    },
    remove_prefix="picongpu_",
)
class Cartesian2DGrid(picmistandard.PICMI_Cartesian2DGrid):
    # number of GPUs to distribute the grid over; whatever form is given, it
    # is normalized to a 2-tuple (see _normalise_n_gpus): a bare int N and [N]
    # both mean "parallelize over N GPUs in y", i.e. (1, N)
    picongpu_n_gpus: Annotated[
        int | Sequence[int] | None,
        BeforeValidator(_reject_bool_n_gpus),
        AfterValidator(lambda x: _normalise_n_gpus(x, 2)),
    ] = Field(default=(1, 1))
    picongpu_grid_dist: None | list[list[int]] = Field(default=None)
    # PIConGPU's 2D setups (e.g. the FoilLCT example) use a <16, 16> super cell.
    picongpu_super_cell_size: tuple[int, int] = Field(default=(16, 16))
    picongpu_absorber_kind: Literal["pml", "exponential"] = Field(default="pml")
    """Absorber profile/kind (``--fieldAbsorber``): ``"pml"`` or ``"exponential"``.

    PIConGPU supports exactly these two profiles (the C++ runtime option
    ``--fieldAbsorber``); both are always compiled in. On an all-periodic grid the
    C++ core overrides the choice and disables the absorber entirely.
    """
    picongpu_pml_cells: None | list[list[int]] = Field(default=None)
    """Per-axis, per-direction absorber depth ``[[negative, positive], ...]`` in cells.

    PIConGPU extension (the standard ``pml_cells`` is per-axis symmetric): exposes the
    full ``NUM_CELLS[3][2]`` matrix so the two boundaries of an axis may differ.
    Mutually exclusive with the standard ``pml_cells``.
    """
    picongpu_exponential_strength: None | list[list[float]] = Field(default=None)
    """Per-axis, per-direction ``exponential::STRENGTH`` ``[[negative, positive], ...]``.

    Only used by the ``"exponential"`` absorber; the default (``1e-3`` everywhere)
    mirrors the static C++ ``fieldAbsorber.param``.
    """
    # In 2D3V the Z cell length (CELL_DEPTH_SI) is the wire-particle integration
    # length used to normalize densities. When left as None, the conversion falls
    # back to the x cell size (dx); set it to override the slab thickness.
    picongpu_cell_depth_si: Annotated[
        float | None,
        AfterValidator(
            lambda x: x if x is None or x > 0 else (_ for _ in ()).throw(ValueError("cell depth must be > 0"))
        ),
    ] = Field(default=None)

    @computed_field
    def picongpu_cell_size(self) -> tuple[int, int]:
        return (
            (self.upper_bound[0] - self.lower_bound[0]) / self.number_of_cells[0],
            (self.upper_bound[1] - self.lower_bound[1]) / self.number_of_cells[1],
        )

    @model_validator(mode="after")
    def _validate(self):
        # A grid must be non-degenerate: every dimension needs at least one cell
        # and a strictly positive extent, otherwise the cell size below would be
        # undefined (ZeroDivisionError) or render an empty C++ grid.
        for dim, name in enumerate(["x", "y"]):
            if self.number_of_cells[dim] < 1:
                raise ValueError(
                    f"number_of_cells[{dim}] ({name} dimension) must be a positive integer. "
                    f"You gave {self.number_of_cells[dim]}."
                )
            if self.upper_bound[dim] <= self.lower_bound[dim]:
                raise ValueError(
                    f"upper_bound in {name} dimension must be greater than lower_bound "
                    f"(got lower={self.lower_bound[dim]}, upper={self.upper_bound[dim]})."
                )
        return self

    def check(self):
        _check_cartesian_grid(self, ["x", "y"])


AnyGrid = Cartesian3DGrid | Cartesian2DGrid
