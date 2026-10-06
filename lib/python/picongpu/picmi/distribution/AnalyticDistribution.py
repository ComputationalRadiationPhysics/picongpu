"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import logging
import traceback
from collections.abc import Callable

import numpy as np
from picmistandard import PICMI_AnalyticDistribution
from picmistandard.base import Expression
from pydantic import ConfigDict, Field, PrivateAttr, computed_field, model_validator
from sympy import Expr, Symbol, lambdify, symbols, sympify

from picongpu.picmi._FieldFunctor import (
    _FieldFunctor,
    callable_extra_parameters,
    expression_from_callable,
    expression_parameter_names,
    expression_string,
    function_from_expression,
)
from picongpu.pypicongpu import species
from picongpu.pypicongpu.util import decorating_class, unsupported

"""
note on rms_velocity:
---------------------
The rms_velocity is converted to a temperature in keV. This conversion requires the mass of the species to be known,
which is not the case inside the picmi density distribution.

As an abstraction, **every** PICMI density distribution implements `picongpu_get_rms_velocity_si()` which returns a
tuple (float, float, float) with the rms_velocity per axis in SI units (m/s).

In case the density profile does not have an rms_velocity, this method **MUST** return (0, 0, 0), which is translated to
"no temperature initialization" by the owning species.

note on drift:
--------------
The drift ("velocity") is represented using either directed_velocity or centroid_velocity (v, gamma*v respectively) and
for the pypicongpu representation stored in a separate object (Drift).

To accommodate that, this separate Drift object can be requested by the method get_picongpu_drift(). In case of no drift,
this method returns None.
"""

#: The coordinate variables shared by the density, momentum and spread fields.
_VARIABLES = ("x", "y", "z")

#: Field families that share the full expression/function/sympy machinery. Each
#: family is one density field or an aligned list of three per-axis fields.
_AXES = ("x", "y", "z")


@decorating_class("density_function", keyword_construction=("density_expression",))
class AnalyticDistribution(PICMI_AnalyticDistribution):
    """
    A plasma whose density, momentum and momentum spread are analytic expressions.

    The class is a thin PICMI-standard wrapper around the shared
    :class:`~picongpu.picmi._FieldFunctor._FieldFunctor`: each of the three
    field families (the density, the per-axis momentum and the per-axis momentum
    spread) is delegated to one or more ``_FieldFunctor`` instances, which carry
    the whole expression/function resolution, parameter collection, validation
    and PMAcc rendering.

    The standard ``density_expression``, ``momentum_expressions`` and
    ``momentum_spread_expressions`` strings are supported; each may instead be
    given as a sympy callable (``density_function``, ``momentum_functions``,
    ``momentum_spread_functions``; the per-axis lists are aligned, ``None`` marks
    an unsupplied axis). The two spellings are interchangeable, both are
    available after construction, and the parsed sympy expressions are exposed as
    the ``density_sympy`` / ``momentum_sympy`` / ``momentum_spread_sympy``
    properties. Provide exactly one of the string or the callable form per field.

    Constants used in any of the expressions (or named as extra parameters of
    any of the callables) may be passed as additional keyword arguments; they are
    collected automatically into ``user_defined_kw`` and substituted before
    rendering, uniformly across all three field families.

    Momentum (``gamma * velocity`` per axis [m/s]) and momentum spread (Gaussian
    thermal sigma per axis [m/s]) are rendered into the constant pypicongpu
    ``Drift`` and ``Temperature`` operations. Position-dependent
    (function of ``x``/``y``/``z``) momentum and spread expressions are not
    implemented yet. Like the PICMI standard, an axis without a momentum
    expression falls back to ``directed_velocity``.

    Writing sympy comes with a few pitfalls but also advantages as listed below.
    Make sure that you familiarise yourself with writing sympy.

    Advantages:
    - The sympy language is closer to mathematical language than to coding
      which might make it more natural to use for some physicists.
    - You can extract the exact distribution
      that was used from the member `density_function`
      and use it any way you'd use any sympy expression.
      In particular, you can print it to various formats,
      say, LaTeX for automated inclusion in papers.
    - We can easily evaluate it from within python.
      This is what the __call__ operator does.
      However, code generation here can have some difficulties
      with advanced broadcasting for numpy variables.
      The operator implements a fallback in such cases.
      This fallback might be slightly slower on large inputs.
      Try rewriting your function, in case you experience performance problems.

    Pitfalls:
    - We don't handle vectors yet.
      But that's probably not too important for density expressions.
      Just be explicit handling multiple vector components for now.
    - Some operations might compile to suboptimal code
      concerning numerical performance and stability.
      Experts might want to inspect the generated C++ code
      and/or check the unit tests for the PMAccPrinter
      to find the precise mapping of sympy expressions
      to PMAcc code.
      Please approach us if you should stumble across this.
    - Control flow is an interesting topic in this regard.
      With respect to your three position coordinates
      (which will be sympy symbols internally),
      you must use pure sympy, e.g.,
      replacing if-conditions with sympy.Piecewise and so on.
      With respect to further parameters,
      you're free to use any python construct you want
      (if-conditions, loops, etc.).
    - sympy.Piecewise has the potentially surprising property
      that any pieces that you leave undefined are interpreted as nan.
      This implies that adding two complementary sympy.Piecewise
      renders the whole expression nan and not -- as you might expect --
      defined on the union of the defined regions.
      There are two options to circumvent this:
      Either you can define multiple sympy.Piecewise with
      (0.0, True) as the last condition which means 0 everywhere else.
      (Make sure it's the last!)
      Summing those up, works just as you'd expect.
      Alternatively, you can define only the (expression, condition) tuples
      and assemble them in a sympy.Piecewise in one go.
      That's the way chosen in the end-to-end tests.

    Parameters:
        density_function (Callable):
            A Python function that takes x, y, z coordinates (in SI units)
            and returns the density (in SI units) at that point.
            It should use sympy functionality.
            Further parameters (beyond x, y and z) are substituted from matching
            keyword arguments given to the constructor or the decorator, exactly
            like the constants collected from a `density_expression`.
            Provide exactly one of `density_function` or `density_expression`.
        density_expression (str):
            A sympy-parseable string expression of the density in terms of
            `x`, `y` and `z` (e.g. `"x*y*z"`). It is string-normalised (mirroring
            the PICMI standard) and then parsed with `sympy.sympify`, so
            non-string inputs are coerced to their string form (e.g. a bare number
            yields a constant density) rather than rejected. It is equivalent to
            the matching `density_function`.
            Provide exactly one of `density_function` or `density_expression`.
        momentum_functions (list of Callable, optional):
            The per-axis sympy callables equivalent to `momentum_expressions`
            (gamma * velocity in SI units, per axis). Each entry takes x, y, z and
            returns the axis expression, or is None for an axis that is not
            supplied. Extra (beyond x, y and z) parameters are substituted from
            matching keyword arguments, exactly like `density_function`.
        momentum_spread_functions (list of Callable, optional):
            The per-axis sympy callables equivalent to
            `momentum_spread_expressions` (Gaussian thermal spread sigma in SI
            units, per axis), with the same per-axis/convention as
            `momentum_functions`.
        directed_velocity (3-tuple of float):
            A collective velocity for the particle distribution, interpreted as a plain velocity.
            Mutually exclusive with ``momentum_expressions``: if either a non-zero
            ``directed_velocity`` and a non-``None`` ``momentum_expressions`` entry are both
            supplied, construction raises.
    """

    # The standard makes density_expression required; PIConGPU additionally allows a
    # sympy based density_function. Both fields are always available after construction:
    # a before validator computes whichever one was not given from the other, and still
    # enforces the standard's "exactly one input" rule for the user-facing construction.
    density_expression: Expression
    density_function: Callable[[Symbol, Symbol, Symbol], Expr]

    # PIConGPU extensions mirroring the density callable for the momentum/spread
    # surface. Each list is aligned per axis with its ``*_expressions`` sibling:
    # a callable in axis ``i`` is the sympy equivalent of the string expression
    # in ``momentum_expressions[i]`` / ``momentum_spread_expressions[i]`` (``None``
    # marks an axis that is not supplied). Exactly as for the density, the string
    # and the callable form are interchangeable and both are available after
    # construction. The parsed sympy expressions are exposed via
    # ``momentum_sympy`` / ``momentum_spread_sympy``.
    momentum_functions: list[Callable[[Symbol, Symbol, Symbol], Expr] | None] = Field(
        default_factory=lambda: [None, None, None]
    )
    momentum_spread_functions: list[Callable[[Symbol, Symbol, Symbol], Expr] | None] = Field(
        default_factory=lambda: [None, None, None]
    )

    _warned_about_lambdify_failure: bool = PrivateAttr(False)

    model_config = ConfigDict(
        arbitrary_types_allowed=True, populate_by_name=True, extra="forbid", validate_assignment=True
    )

    # ------------------------------------------------------------------
    # Binding and delegation to _FieldFunctor
    # ------------------------------------------------------------------

    def _axis_functors(self, field: str) -> list[_FieldFunctor | None]:
        # Both spellings are populated after construction, so prefer the string
        # (the family resolver has already checked that the callable agrees).
        expressions = getattr(self, f"{field}_expressions")
        functions = getattr(self, f"{field}_functions")
        functors: list[_FieldFunctor | None] = []
        for index, axis in enumerate(_AXES):
            if expressions[index] is None and functions[index] is None:
                functors.append(None)
                continue
            functors.append(
                _FieldFunctor(
                    expression=expressions[index],
                    function=None if expressions[index] is not None else functions[index],
                    variables=_VARIABLES,
                    parameters=self.user_defined_kw,
                    context=f"AnalyticDistribution {field} {axis}",
                )
            )
        return functors

    @property
    def _density_functor(self) -> _FieldFunctor:
        # Use the callable: it is the original user input for the callable path and
        # an exact wrapper around the parsed string for the expression path, so the
        # density is not distorted by a string round trip.
        return _FieldFunctor(
            function=self.density_function,
            variables=_VARIABLES,
            parameters=self.user_defined_kw,
            context="AnalyticDistribution density",
        )

    # ------------------------------------------------------------------
    # Input resolution: exactly one spelling, and both always populated
    # ------------------------------------------------------------------

    @classmethod
    def _callable_expression(cls, function, user_defined_kw) -> Expr:
        """The sympy expression of a callable field, parameters substituted.

        Uses the shared callable-resolution primitive directly (rather than a
        full :class:`_FieldFunctor`) so that an invalid callable is only
        rejected when the field is actually rendered, preserving the
        pre-existing lazy behaviour of the density input resolution.
        """
        coordinates = {name: symbols(name) for name in _VARIABLES}
        return expression_from_callable(function, coordinates, user_defined_kw).subs(user_defined_kw)

    @classmethod
    def _bind_function(cls, function, user_defined_kw):
        """A field callable with its extra (beyond x, y, z) parameters bound.

        The returned ``g(x, y, z)`` yields the sympy expression for that field;
        it is the serial fallback of :meth:`__call__` when sympy's numpy code
        generation cannot handle the broadcasting.
        """
        expression = cls._callable_expression(function, user_defined_kw)
        coordinates = {name: symbols(name) for name in _VARIABLES}
        return lambda x, y, z: expression.subs(dict(zip(coordinates.values(), (x, y, z))))

    @classmethod
    def _collect_callable_user_defined_kw(cls, data, function, consumed: set[str]) -> None:
        """Register the extra keyword arguments a callable field asks for.

        ``consumed`` is the set of data keys already claimed by another field's
        callable, so a parameter shared by several fields is collected once.
        Unknown keywords are still rejected by the ``extra="forbid"`` config.
        """
        if function is None:
            return
        user_defined_kw = dict(data.get("user_defined_kw", {}))
        for name in sorted(callable_extra_parameters(function)):
            if name in data and name not in consumed:
                user_defined_kw[name] = data.pop(name)
                consumed.add(name)
        if user_defined_kw:
            data["user_defined_kw"] = user_defined_kw

    @classmethod
    def _collect_expression_user_defined_kw(cls, data, expression, consumed: set[str]) -> None:
        """Register the free symbols an expression string references.

        The PICMI-standard collector only scans the density and momentum
        expressions; calling it per family (or on a callable's derived string)
        makes the ``user_defined_kw`` mechanism uniform across all three families.
        """
        if expression is None:
            return
        user_defined_kw = dict(data.get("user_defined_kw", {}))
        for name in sorted(expression_parameter_names(expression, _VARIABLES)):
            if name in data and name not in consumed:
                user_defined_kw[name] = data.pop(name)
                consumed.add(name)
        if user_defined_kw:
            data["user_defined_kw"] = user_defined_kw

    @classmethod
    def _resolve_family(cls, data, field: str, consumed: set[str]) -> None:
        """Bring the string and callable spelling of one field family in sync.

        Either input is accepted, the family's ``_expression``s are recomputed
        from the callables so that both are always available, and the parameters
        each form needs are collected. Supplying both spellings for the same
        field is rejected.
        """
        expressions_key = f"{field}_expressions"
        functions_key = f"{field}_functions"
        expressions = data.get(expressions_key)
        functions = data.get(functions_key)
        if expressions is None and functions is None:
            return
        expressions = list(expressions) if expressions is not None else [None] * len(_AXES)
        functions = list(functions) if functions is not None else [None] * len(_AXES)
        if len(expressions) != len(_AXES) or len(functions) != len(_AXES):
            raise ValueError(f"{field} must have exactly {len(_AXES)} entries (one per axis).")

        resolved_expressions: list[str | None] = []
        resolved_functions: list[Callable | None] = []
        for index, axis in enumerate(_AXES):
            expression = expressions[index]
            function = functions[index]
            if expression is None and function is None:
                resolved_expressions.append(None)
                resolved_functions.append(None)
                continue
            cls._collect_expression_user_defined_kw(data, expression, consumed)
            cls._collect_callable_user_defined_kw(data, function, consumed)
            # One shared _FieldFunctor per axis owns the whole resolution: it
            # accepts either spelling, validates that two given together agree
            # and translates to the canonical string/callable pair.
            functor = _FieldFunctor(
                expression=expression,
                function=function,
                variables=_VARIABLES,
                parameters=data.get("user_defined_kw") or {},
                context=f"AnalyticDistribution {field} {axis}",
            )
            resolved_expressions.append(functor.expression)
            resolved_functions.append(functor.function)
        data[expressions_key] = resolved_expressions
        data[functions_key] = resolved_functions

    @classmethod
    def _reject_conflicting_drift(cls, data) -> None:
        # directed_velocity (plain velocity) and momentum_expressions (gamma * velocity) are
        # two different, mutually exclusive ways of setting a drift. A non-zero directed_velocity
        # combined with a non-None momentum expression was previously silently discarded; reject
        # the ambiguous combination so the two can't silently override each other.
        directed_velocity = data.get("directed_velocity")
        if directed_velocity is None:
            directed_velocity = (0.0, 0.0, 0.0)
        momentum_expressions = data.get("momentum_expressions")
        if momentum_expressions is None:
            momentum_expressions = [None, None, None]
        has_directed = any(float(v) != 0.0 for v in directed_velocity)
        has_momentum = any(e is not None for e in momentum_expressions)
        if has_directed and has_momentum:
            raise ValueError(
                "directed_velocity and momentum_expressions are mutually exclusive; "
                "provide exactly one of them to set the drift."
            )

    @model_validator(mode="before")
    @classmethod
    def _resolve_inputs(cls, data, info):
        # With ``validate_assignment=True`` (inherited from the standard base class)
        # every assignment re-enters this validator with *both* density fields already
        # populated, so the "exactly one input" rule below must not fire here. Keep the
        # two density fields consistent when one of them is assigned, and let the field
        # validators handle any other assignment unchanged.
        if not isinstance(data, dict):
            return data
        data = dict(data)
        if info.field_name is not None:
            if info.field_name == "density_expression" and data.get("density_expression") is not None:
                data["density_function"] = function_from_expression(
                    sympify(f"{data['density_expression']}".replace("\n", "")), _VARIABLES
                )
            elif info.field_name == "density_function" and data.get("density_function") is not None:
                data["density_expression"] = expression_string(
                    cls._callable_expression(data["density_function"], data.get("user_defined_kw") or {})
                )
            return data

        has_function = data.get("density_function") is not None
        has_expression = data.get("density_expression") is not None
        if has_function == has_expression:
            raise ValueError("exactly one of density_function or density_expression must be provided")

        consumed: set[str] = set()
        cls._resolve_family(data, "momentum", consumed)
        cls._resolve_family(data, "momentum_spread", consumed)
        cls._reject_conflicting_drift(data)
        if has_expression:
            cls._collect_expression_user_defined_kw(data, data["density_expression"], consumed)
            data["density_function"] = function_from_expression(
                sympify(f"{data['density_expression']}".replace("\n", "")), _VARIABLES
            )
        else:
            cls._collect_callable_user_defined_kw(data, data["density_function"], consumed)
            data["density_expression"] = expression_string(
                cls._callable_expression(data["density_function"], data.get("user_defined_kw") or {})
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _resolve_axis_functions(cls, data, info):
        # ``validate_assignment=True`` re-enters the nested list field validator
        # on every direct assignment. Bring the assigned axis's ``*_expressions``
        # and ``*_functions`` counterparts back in sync, exactly like the density
        # fields, so the two spellings stay interchangeable per axis.
        field = info.field_name
        pairs = {
            "momentum_expressions": "momentum",
            "momentum_functions": "momentum",
            "momentum_spread_expressions": "momentum_spread",
            "momentum_spread_functions": "momentum_spread",
        }
        if field not in pairs or not isinstance(data, dict):
            return data
        data = dict(data)
        assigned = data.get(field)
        if assigned is None:
            return data
        # Drop the *other* spelling's stale value and recompute it from what is
        # being assigned, so a direct assignment cannot leave the pair inconsistent.
        family = pairs[field]
        other = f"{family}_functions" if field == f"{family}_expressions" else f"{family}_expressions"
        data.pop(other, None)
        cls._resolve_family(data, family, set())
        return data

    # ------------------------------------------------------------------
    # Public sympy views
    # ------------------------------------------------------------------

    def _density_expression(self) -> Expr:
        # The ``user_defined_kw`` constants are substituted here, mirroring the
        # public ``density_sympy`` view; the per-axis views substitute too, so the
        # equivalent spellings compare equal.
        x, y, z = symbols("x,y,z")
        return self._density_functor.sympy + (0 * x * y * z)

    @computed_field
    @property
    def density_sympy(self) -> Expr:
        """The density as a sympy expression of x, y and z (public counterpart of ``density_function``)."""
        return self._density_expression()

    def _axis_sympy(self, field: str) -> list[Expr | None]:
        return [None if functor is None else functor.sympy for functor in self._axis_functors(field)]

    @computed_field
    @property
    def momentum_sympy(self) -> list[Expr | None]:
        """The per-axis momentum expressions as sympy expressions (constants substituted).

        Public counterpart of ``momentum_functions``; ``None`` marks an axis that
        was not supplied. The ``user_defined_kw`` constants are substituted, just
        like in ``density_sympy``.
        """
        return self._axis_sympy("momentum")

    @computed_field
    @property
    def momentum_spread_sympy(self) -> list[Expr | None]:
        """The per-axis thermal spread expressions as sympy expressions (constants substituted).

        Public counterpart of ``momentum_spread_functions``; ``None`` marks an axis
        that was not supplied. The ``user_defined_kw`` constants are substituted,
        just like in ``density_sympy``.
        """
        return self._axis_sympy("momentum_spread")

    @property
    def dim(self) -> int:
        """The number of spatial dimensions the density depends on (2 or 3)."""
        z = Symbol("z")
        return 2 if z not in self.density_sympy.free_symbols else 3

    # ------------------------------------------------------------------
    # pypicongpu translation
    # ------------------------------------------------------------------

    def get_as_pypicongpu(self, _):
        unsupported("fill in", self.fill_in)
        unsupported("lower bound", self.lower_bound, [None, None, None])
        unsupported("upper bound", self.upper_bound, [None, None, None])
        return species.operation.densityprofile.FreeFormula(density_expression=self._density_functor.sympy)

    def picongpu_get_rms_velocity_si(self) -> tuple[float, float, float]:
        rms_velocity = [float(v) for v in self.rms_velocity]
        return tuple(map(lambda r, s: max(r, s), rms_velocity, self._constant_momentum_spread_si()))

    def get_picongpu_drift(self) -> species.operation.momentum.Drift | None:
        """
        Get drift for pypicongpu
        :return: pypicongpu drift object or None
        """
        # The legacy directed_velocity is a plain velocity (from_velocity); the standard
        # momentum_expressions are gamma * velocity (from_gamma_velocity).
        if any(v != 0 for v in self.directed_velocity):
            return species.operation.momentum.Drift.from_velocity(tuple(self.directed_velocity))  # type: ignore[arg-type]
        gamma_velocity = self._constant_gamma_velocity()
        if gamma_velocity is None:
            return None
        return species.operation.momentum.Drift.from_gamma_velocity(gamma_velocity)

    def _constant_expression(self, field: str, expression: Expr) -> float:
        """
        Evaluate a constant momentum/spread expression (after substituting user_defined_kw)
        to a plain float. Position-dependent expressions (still referencing x/y/z) are
        rejected, and so are expressions whose parameters were never supplied.
        """
        x, y, z = symbols("x,y,z")
        resolved = sympify(expression).subs(self.user_defined_kw)
        if not resolved.free_symbols <= {x, y, z}:
            missing = sorted(sym.name for sym in resolved.free_symbols - {x, y, z})
            raise ValueError(f"{field} must be constant, but {expression!r} is missing a value for {missing}.")
        if resolved.free_symbols:
            unsupported(f"position-dependent {field}", expression)
        return float(resolved)

    def _constant_gamma_velocity(self) -> tuple[float, float, float] | None:
        """
        Evaluate the constant momentum_expressions (gamma * velocity per axis [m/s]) into a
        3-tuple. Any axis whose expression is None contributes zero (the directed_velocity is
        handled separately, using plain velocity semantics).

        Returns None if every resolved axis is zero (no drift).
        """
        gamma_velocity = [
            0.0 if functor is None else self._constant_expression("momentum_expressions", functor.symbolic)
            for functor in self._axis_functors("momentum")
        ]
        if np.allclose(gamma_velocity, 0.0):
            return None
        return tuple(gamma_velocity)  # type: ignore[return-value]

    def _constant_momentum_spread_si(self) -> tuple[float, float, float]:
        """
        Evaluate the constant momentum_spread_expressions (Gaussian sigma per axis [m/s]).
        Any axis whose expression is None contributes zero.
        """
        return tuple(
            0.0 if functor is None else self._constant_expression("momentum_spread_expressions", functor.symbolic)
            for functor in self._axis_functors("momentum_spread")
        )

    # ------------------------------------------------------------------
    # Python evaluation of the density
    # ------------------------------------------------------------------

    def __call__(self, *args, **kwargs):
        args = tuple(np.asarray(a) for a in args)
        expression = self._density_functor.sympy
        try:
            # This produces faster code but the code generation is not perfect.
            # There are cases where the generated code can't handle broadcasting properly.
            return lambdify(symbols("x,y,z"), expression, "numpy")(*args, **kwargs)
        # We explicitly want this to be as broad as possible
        # because we have a second shot.
        # There should be no instances of this being dangerous during idiomatic use of this functionality.
        except Exception:
            if not self._warned_about_lambdify_failure:
                message = (
                    "Sympy did not manage to produce proper numpy code for your AnalyticDistribution. "
                    "If you run into performance problems, try to rewrite your function. "
                    "Here's the original error message:"
                )
                logging.warning(message)
                logging.warning(traceback.format_exc())
                logging.warning("Continuing operation using a slower serialised version now.")
                self._warned_about_lambdify_failure = True
        # This basically calls the original function in a big loop.
        # Slower but more reliable in some cases of difficult broadcasting.
        return np.vectorize(self._bind_function(self.density_function, self.user_defined_kw))(*args, **kwargs)

    # ------------------------------------------------------------------
    # Semantic equality
    # ------------------------------------------------------------------

    def _equality_key(self):
        """Semantic identity: the rendered density plus the standard surface.

        The density_function is deliberately excluded, so the equivalent spellings
        (decorator with constants, density_expression string, density_function
        callable) compare equal.
        """
        return (
            self._density_expression(),
            tuple(self.momentum_sympy),
            tuple(self.momentum_spread_sympy),
            tuple(self.rms_velocity),
            tuple(self.directed_velocity),
            tuple(self.lower_bound),
            tuple(self.upper_bound),
            self.fill_in,
        )

    def __eq__(self, other):
        if not isinstance(other, AnalyticDistribution):
            return NotImplemented
        return self._equality_key() == other._equality_key()

    def __hash__(self):
        return hash(self._equality_key())
