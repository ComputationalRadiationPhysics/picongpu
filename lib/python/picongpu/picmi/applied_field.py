"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from collections.abc import Callable

import numpy as np
import sympy
from picmistandard import PICMI_AnalyticAppliedField, PICMI_ConstantAppliedField
from picmistandard.base import Expression
from pydantic import BaseModel, ConfigDict, Field, model_validator

from picongpu.picmi._FieldFunctor import (
    _FieldFunctor,
    callable_extra_parameters,
    expression_parameter_names,
)
from picongpu.pypicongpu import util
from picongpu.pypicongpu._field_functor import (
    check_allowed_symbols,
    check_parameter_names,
    sympify_expression,
)
from picongpu.pypicongpu.backgroundfield import BackgroundField

#: Component keys shared by the pypicongpu background field and the PICMI fields.
COMPONENTS = ("Ex", "Ey", "Ez", "Bx", "By", "Bz")

_ANALYTIC_FREE_VARIABLES = ("x", "y", "z", "t")


class _InfluenceOptions(BaseModel):
    """
    PIConGPU-specific visibility knobs shared by the applied-field classes.

    The PICMI standard has no notion of field-background visibility, so these
    are PIConGPU extensions. They are declared as real fields (rather than left
    to the standard ``user_defined_kw`` catch-all) so that they are not silently
    treated as expression parameters. The names carry the ``picongpu_`` prefix
    as required for code-specific PICMI inputs.

    The background is always applied to the grid around the particle push, so
    the particles always feel it; these knobs only control whether plugins and
    dumps see it.
    """

    picongpu_influences_plugins: bool = Field(
        default=True,
        description="Whether plugins see the background (C++ ``fieldBackground.influencesPlugins``).",
    )
    picongpu_influences_dumps: bool = Field(
        default=True,
        description="Whether dumps (incl. checkpoints) include the background (C++ ``fieldBackground.influencesDumps``).",
    )


def _check_only_full_domain(applied_field) -> None:
    """
    Reject region-restricted applied fields for now.

    The C++ field-background path currently renders the functor over the whole
    simulation domain. Support for PICMI ``lower_bound``/``upper_bound``
    regions is planned, but not yet implemented, so fail loudly instead of
    silently applying the field everywhere.
    """
    for bound in (applied_field.lower_bound, applied_field.upper_bound):
        if any(component is not None for component in (bound or [])):
            util.unsupported("applied-field region restriction (lower_bound/upper_bound)", bound)


def _influence_kwargs(applied_field) -> dict:
    return dict(
        influences_plugins=applied_field.picongpu_influences_plugins,
        influences_dumps=applied_field.picongpu_influences_dumps,
    )


def merge_influence(applied_fields) -> dict:
    """
    Combine the visibility knobs of several applied fields.

    The knobs configure the *single* C++ ``FieldBackgroundE``/``FieldBackgroundB``
    pair, so all applied fields must agree on them.
    """
    merged = None
    for applied_field in applied_fields:
        influence = _influence_kwargs(applied_field)
        if merged is None:
            merged = influence
        elif merged != influence:
            util.unsupported("applied fields with conflicting influence knobs", influence)
    return merged


def _check_expression_symbols(expressions: dict[str, sympy.Expr], parameters: list[dict]) -> None:
    allowed = set(_ANALYTIC_FREE_VARIABLES) | {parameter["name"] for parameter in parameters}
    check_allowed_symbols(expressions, allowed, "AnalyticAppliedField")


def _validate_components(expressions: dict[str, sympy.Expr], parameters: list[dict]) -> None:
    """
    Run the expression checks shared by the direct and the combined translation paths.

    Both :meth:`AnalyticAppliedField.get_as_pypicongpu` and
    :func:`combine_applied_fields` (the ``Simulation`` path) must reject undefined
    symbols and non-renderable parameter names, otherwise invalid C++ is emitted
    and only fails at device-compile time.
    """
    check_parameter_names(parameter["name"] for parameter in parameters)
    _check_expression_symbols(
        {component: expression for component, expression in expressions.items() if expression is not None},
        parameters,
    )


def combine_applied_fields(applied_fields) -> BackgroundField:
    """
    Combine several applied fields into the single pypicongpu background field.

    The C++ core evaluates a single ``FieldBackgroundE``/``FieldBackgroundB``
    functor pair, so the individual E/B contributions are summed per component
    (constants and expressions alike) and the parameters are merged.
    """
    applied_fields = list(applied_fields)
    influence = merge_influence(applied_fields)
    combined: dict[str, sympy.Expr] = {component: sympy.Integer(0) for component in COMPONENTS}
    parameters: dict[str, float] = {}
    for applied_field in applied_fields:
        _check_only_full_domain(applied_field)
        for component, expression in applied_field.get_components().items():
            if expression is not None:
                combined[component] = combined[component] + expression
        for parameter in applied_field.get_parameters():
            name, value = parameter["name"], parameter["value"]
            if name in parameters and parameters[name] != value:
                util.unsupported(f"redefining parameter {name!r} with a different value", value)
            parameters[name] = value
    parameter_list = [{"name": name, "value": value} for name, value in sorted(parameters.items())]
    _validate_components(combined, parameter_list)
    return BackgroundField(
        **{component.lower(): expression for component, expression in combined.items()},
        user_defined_kw=parameter_list,
        **influence,
    )


class ConstantAppliedField(_InfluenceOptions, PICMI_ConstantAppliedField):
    """
    PIConGPU implementation of the PICMI ``ConstantAppliedField``.

    A constant field is added to the grid E and B fields around the particle
    push: particles feel it, but the field solver does not evolve it (see the
    C++ ``fieldBackground.param`` + ``FieldBackground.hpp``).

    Only whole-domain fields are supported so far, so ``lower_bound`` and
    ``upper_bound`` must be left as their default (all ``None``).

    The standard PICMI attribute names (``Ex``, ``Ey``, ``Ez`` in V/m and
    ``Bx``, ``By``, ``Bz`` in T) are used verbatim.
    """

    def get_components(self) -> dict[str, sympy.Expr | None]:
        """The constant E/B components as sympy expressions (``None`` is zero)."""
        return {
            component: None if getattr(self, component) is None else sympify_expression(getattr(self, component))
            for component in COMPONENTS
        }

    def __call__(self, x=0.0, y=0.0, z=0.0, t=0.0) -> dict[str, object]:
        """
        The six field components (in SI units) at the given coordinates.

        A constant field does not depend on position or time; the arguments are
        accepted and ignored so that constant and analytic applied fields share
        one call-operator interface. Unset components are ``None``.
        """
        return {
            component: None if expression is None else float(expression)
            for component, expression in self.get_components().items()
        }

    def get_parameters(self) -> list[dict]:
        return []

    def get_as_pypicongpu(self) -> BackgroundField:
        _check_only_full_domain(self)
        return BackgroundField(
            **{component.lower(): expression for component, expression in self.get_components().items()},
            **_influence_kwargs(self),
        )


class AnalyticAppliedField(_InfluenceOptions, PICMI_AnalyticAppliedField):
    """
    PIConGPU implementation of the PICMI ``AnalyticAppliedField``.

    An analytic field is added to the grid E and B fields around the particle
    push: particles feel it, but the field solver does not evolve it (see the
    C++ ``fieldBackground.param`` + ``FieldBackground.hpp``).

    The field mirrors the interface of
    :class:`~picongpu.picmi.distribution.AnalyticDistribution.AnalyticDistribution`
    (see :doc:`/python_package/selected_topics/functors`): each of the six
    components is one shared
    :class:`~picongpu.picmi._FieldFunctor._FieldFunctor` and is exposed in all
    three interchangeable spellings:

    * ``<component>_expression`` -- a sympy-parseable string of ``x``, ``y``,
      ``z`` and ``t``,
    * ``<component>_function`` -- a callable of ``x``, ``y``, ``z`` and ``t``,
    * ``<component>_sympy`` -- the resolved :class:`sympy.Expr`.

    Any one of them may be supplied; the others are computed from it, so all
    three are available and consistent after construction. Supplying several
    spellings for one component is allowed as long as they agree, otherwise it
    is rejected. Named parameters used by the spellings are supplied as
    additional keyword arguments and rendered as compile-time constants.

    ``Ex`` etc. are in V/m and ``Bx`` etc. in T. Only whole-domain fields are
    supported so far, so ``lower_bound`` and ``upper_bound`` must be left as
    their default (all ``None``).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # The full triple is declared explicitly on the PIConGPU subclass (the
    # ``*_expression`` fields would otherwise only be inherited from the PICMI
    # standard base), so that all three spellings are owned here and backed by
    # the same shared :class:`_FieldFunctor`.
    Ex_expression: Expression | None = None
    Ey_expression: Expression | None = None
    Ez_expression: Expression | None = None
    Bx_expression: Expression | None = None
    By_expression: Expression | None = None
    Bz_expression: Expression | None = None
    Ex_function: Callable | None = None
    Ey_function: Callable | None = None
    Ez_function: Callable | None = None
    Bx_function: Callable | None = None
    By_function: Callable | None = None
    Bz_function: Callable | None = None
    Ex_sympy: sympy.Expr | None = None
    Ey_sympy: sympy.Expr | None = None
    Ez_sympy: sympy.Expr | None = None
    Bx_sympy: sympy.Expr | None = None
    By_sympy: sympy.Expr | None = None
    Bz_sympy: sympy.Expr | None = None

    # ------------------------------------------------------------------
    # Input resolution: every component is one _FieldFunctor
    # ------------------------------------------------------------------

    @classmethod
    def _collect_expression_user_defined_kw(cls, data, expression, consumed: set[str]) -> None:
        """Register the free symbols an expression string references."""
        if expression is None:
            return
        user_defined_kw = dict(data.get("user_defined_kw") or {})
        for name in sorted(expression_parameter_names(expression, _ANALYTIC_FREE_VARIABLES)):
            if name in data and name not in consumed:
                user_defined_kw[name] = data.pop(name)
                consumed.add(name)
        if user_defined_kw:
            data["user_defined_kw"] = user_defined_kw

    @classmethod
    def _collect_callable_user_defined_kw(cls, data, function, consumed: set[str]) -> None:
        """Register the extra keyword arguments a callable field asks for.

        Only kwargs whose names are actually declared as extra parameters of the
        callable are registered; unknown ones are still rejected by the
        ``extra="forbid"`` config (so a typo does not silently become an unused
        constant).
        """
        if function is None:
            return
        user_defined_kw = dict(data.get("user_defined_kw") or {})
        for name in sorted(callable_extra_parameters(function, _ANALYTIC_FREE_VARIABLES)):
            if name in data and name not in consumed:
                user_defined_kw[name] = data.pop(name)
                consumed.add(name)
        if user_defined_kw:
            data["user_defined_kw"] = user_defined_kw

    @staticmethod
    def _component_spellings(component: str) -> tuple[str, str, str]:
        return (f"{component}_expression", f"{component}_function", f"{component}_sympy")

    @classmethod
    def _resolve_component(cls, data, component: str, consumed: set[str]) -> None:
        """Bring the three spellings of one component in sync via one ``_FieldFunctor``.

        Any of the string, callable or sympy spelling is accepted, the missing
        ones are computed from it, and the named parameters each spelling needs
        are collected. The shared :class:`_FieldFunctor` validates that
        spellings given together agree.
        """
        expression = data.get(f"{component}_expression")
        function = data.get(f"{component}_function")
        sympy_expression = data.get(f"{component}_sympy")
        if expression is None and function is None and sympy_expression is None:
            return
        # The sympy spelling is scanned like the string spelling (both are
        # rendered to a Python string form), so parameters referenced only in
        # ``<component>_sympy`` are collected as well.
        cls._collect_expression_user_defined_kw(data, expression, consumed)
        cls._collect_expression_user_defined_kw(data, sympy_expression, consumed)
        cls._collect_callable_user_defined_kw(data, function, consumed)
        functor = _FieldFunctor(
            expression=expression,
            function=function,
            sympy_expression=sympy_expression,
            variables=_ANALYTIC_FREE_VARIABLES,
            parameters=data.get("user_defined_kw") or {},
            context=f"AnalyticAppliedField {component}",
        )
        data[f"{component}_expression"] = functor.expression
        data[f"{component}_function"] = functor.function
        data[f"{component}_sympy"] = functor.sympy

    @model_validator(mode="before")
    @classmethod
    def _resolve_components(cls, data, info):
        # With ``validate_assignment=True`` (inherited from the standard base
        # class) every assignment re-enters this validator with all fields
        # already populated. The assigned field is the new source of truth, so
        # drop its two counterparts before re-resolving; all other components
        # are re-validated unchanged.
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if info.field_name is not None:
            for component in COMPONENTS:
                spellings = cls._component_spellings(component)
                if info.field_name in spellings:
                    for spelling in spellings:
                        if spelling != info.field_name:
                            data.pop(spelling, None)
        consumed: set[str] = set()
        for component in COMPONENTS:
            cls._resolve_component(data, component, consumed)
        return data

    # ------------------------------------------------------------------
    # Delegation to the shared functor
    # ------------------------------------------------------------------

    def _component_functor(self, component: str) -> _FieldFunctor | None:
        expression = getattr(self, f"{component}_expression")
        function = getattr(self, f"{component}_function")
        if expression is None and function is None:
            return None
        return _FieldFunctor(
            expression=expression,
            function=function,
            variables=_ANALYTIC_FREE_VARIABLES,
            parameters=self.user_defined_kw,
            context=f"AnalyticAppliedField {component}",
        )

    def __call__(self, x=0.0, y=0.0, z=0.0, t=0.0) -> dict[str, object]:
        """
        Evaluate all six field components (in SI units) at the given coordinates.

        Mirrors :meth:`~picongpu.picmi.distribution.AnalyticDistribution.AnalyticDistribution.__call__`:
        the coordinates may be scalars or numpy arrays and are broadcast against
        each other. Returns a mapping from the PIConGPU component keys
        (``"Ex"`` ... ``"Bz"``) to the evaluated values; an unset component is
        ``None``. Named parameters are already substituted, exactly as in the
        rendered C++ functor.
        """
        coordinates = np.broadcast_arrays(*(np.asarray(argument) for argument in (x, y, z, t)))
        return {
            component: (
                None if (functor := self._component_functor(component)) is None else functor.evaluate(*coordinates)
            )
            for component in COMPONENTS
        }

    def get_components(self) -> dict[str, sympy.Expr | None]:
        """The E/B components as sympy expressions with parameters kept symbolic."""
        return {
            component: None if (functor := self._component_functor(component)) is None else functor.symbolic
            for component in COMPONENTS
        }

    def get_parameters(self) -> list[dict]:
        return [{"name": name, "value": value} for name, value in sorted(self.user_defined_kw.items())]

    def get_as_pypicongpu(self) -> BackgroundField:
        _check_only_full_domain(self)
        parameters = self.get_parameters()
        components = self.get_components()
        _validate_components(components, parameters)
        return BackgroundField(
            **{component.lower(): expression for component, expression in components.items()},
            user_defined_kw=parameters,
            **_influence_kwargs(self),
        )


AnyAppliedField = ConstantAppliedField | AnalyticAppliedField

__all__ = ["AnalyticAppliedField", "AnyAppliedField", "ConstantAppliedField"]
