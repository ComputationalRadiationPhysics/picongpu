"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

The shared, sympy-backed field functor of the PICMI layer.

Every settable field of the analytic PICMI classes (the six applied-field
components of :class:`~picongpu.picmi.applied_field.AnalyticAppliedField`, and
the density / per-axis momentum / momentum-spread fields of
:class:`~picongpu.picmi.distribution.AnalyticDistribution.AnalyticDistribution`)
is described by the same three interchangeable spellings:

* ``<field>_function`` -- a Python callable of the coordinate variables,
* ``<field>_expression`` -- a sympy-parseable string,
* ``<field>_sympy`` -- the resolved :class:`sympy.Expr`.

:class:`_FieldFunctor` backs such a triple: it accepts any of the spellings,
validates that they agree, and translates between them. It also collects and
validates the named parameters supplied as additional keyword arguments
(``user_defined_kw``) and renders the expression to PMAcc C++.

The rendering primitives (the :class:`PMAccPrinter` wrapper, identifier
escaping and the symbol checks) live in
:mod:`picongpu.pypicongpu._field_functor` because they are shared with the
pypicongpu models; this module only adds the PICMI-level triple.
"""

import inspect
import re
from collections.abc import Callable, Iterable, Mapping

import numpy as np
import sympy
from sympy.printing.str import StrPrinter

from picongpu.pypicongpu._field_functor import (
    check_allowed_symbols,
    check_parameter_names,
    render,
    render_identifier,
    sympify_expression,
)


class _LosslessFloatPrinter(StrPrinter):
    """A string printer that emits every ``Float`` losslessly.

    sympy's default ``sstr`` rounds floating-point numbers to a fixed number of
    significant digits, so a value such as ``0.6e-3 + 5.0e-4`` (which is not
    exactly representable) prints as ``0.0011`` and re-parses to a *different*
    double, one ULP away.  Printing ``repr(float(...))`` instead yields the
    shortest decimal that round-trips exactly to the same 64-bit float, so the
    canonical ``*_expression`` string is a faithful representation of the
    symbolic expression.
    """

    def _print_Float(self, expr):
        return repr(float(expr))


def expression_string(expression) -> str:
    """
    The canonical PICMI string spelling of a sympy expression.

    This is the inverse of :func:`sympify_expression`, used when a model must
    expose the ``*_expression`` string field for an expression that was actually
    supplied as a callable: string-normalised (newlines removed), stable across a
    sympify round trip, and -- unlike sympy's default ``sstr`` -- lossless for
    floating-point numbers.
    """
    printer = _LosslessFloatPrinter()
    printer._settings["order"] = "none"
    return printer.doprint(sympy.sympify(expression)).replace("\n", "")


def _canonical_float64(expression) -> sympy.Expr:
    """Rebuild every ``Float`` atom from its 64-bit value at a fixed precision.

    Two spellings of the same field can carry the same float64 value at
    different sympy precisions (e.g. a precision-53 float from a Python literal
    vs. a precision-60 float from re-parsing a decimal string).  Normalising them
    to a common representation lets the agreement check treat them as equal
    instead of failing on a sub-ULP precision artifact.
    """
    return expression.xreplace(
        {value: sympy.Float(repr(float(value)), precision=53) for value in expression.atoms(sympy.Float)}
    )


def function_from_expression(expression, variables: Iterable[str] = ("x", "y", "z")) -> Callable:
    """
    The callable spelling of a sympy expression.

    The returned function takes the coordinate variables (in the order of
    ``variables``) and substitutes them into the expression, so it is the
    callable counterpart of a ``*_expression`` string.
    """
    variables = tuple(variables)
    symbols = tuple(sympy.Symbol(name) for name in variables)
    resolved = sympy.sympify(expression)
    return lambda *args: resolved.subs(dict(zip(symbols, args)))


def _accepted_parameters(function: Callable) -> set[str] | None:
    """
    Names the callable accepts as keyword arguments, or ``None`` for ``**kwargs``.

    Returns ``None`` when the callable accepts arbitrary keyword arguments, in
    which case every parameter may be passed. Returns an empty set when the
    signature cannot be inspected (e.g. some C callables).
    """
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return set()
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return None
    return {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }


def callable_parameter_names(function: Callable) -> set[str] | None:
    """
    Names the callable accepts beyond the coordinate/time variables.

    Returns ``None`` for a callable with ``**kwargs`` (any keyword may be a
    parameter), an empty set when the signature cannot be inspected.
    """
    accepted = _accepted_parameters(function)
    if accepted is None:
        return None
    return accepted


def callable_extra_parameters(function: Callable, variables: Iterable[str] = ("x", "y", "z")) -> set[str]:
    """
    Names a callable accepts beyond the coordinate variables.

    A plain ``f(x, y, z)`` yields the empty set and ``f(x, y, z, a, b)`` yields
    ``{"a", "b"}``. Only explicitly named parameters (positional-or-keyword and
    keyword-only) are reported; a catch-all ``**kwargs`` is ignored, so the
    caller cannot silently bind an arbitrary keyword to it. Returns an empty set
    when the signature cannot be inspected.
    """
    accepted = _accepted_parameters(function)
    if accepted is None:
        return set()
    return accepted - set(variables)


def expression_parameter_names(expression, variables: Iterable[str] = ("x", "y", "z")) -> set[str]:
    """
    The identifier names used by an expression beyond the coordinate variables.

    Extracted with a word-boundary scan (the same mechanism as the PICMI
    standard's parameter collector), so a name that also exists in sympy's
    namespace (e.g. ``E1``) is still recognised as a candidate parameter. The
    caller only collects the names that were actually supplied as keyword
    arguments, so unknown identifiers are left to the
    :class:`_FieldFunctor` symbol check.
    """
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", f"{expression}"))
    return identifiers - set(variables)


def expression_from_callable(
    function: Callable,
    variables: Mapping[str, sympy.Symbol],
    parameters: Mapping[str, float] | None = None,
) -> sympy.Expr:
    """
    Evaluate a user-supplied callable on the coordinate symbols.

    The callable is called with the free variables (in the order given by
    ``variables``) and must return something sympy can understand. Additional
    named parameters are passed as keyword arguments to the parameters the
    callable actually asks for (the same additional-kwargs mechanism as for
    expression strings).
    """
    # Named parameters are passed as sympy symbols (not their numeric values) so
    # that the resulting expression stays symbolic and the values are rendered
    # as compile-time constants by pypicongpu, exactly like ``*_expression``
    # strings with additional keyword arguments. We only pass the parameters
    # the callable actually asks for, so a function that only uses some
    # coordinates keeps working when unrelated parameters are present.
    accepted = _accepted_parameters(function)
    names = list(parameters or {})
    arguments = dict(variables)
    for name in names:
        if accepted is None or name in accepted:
            arguments[name] = sympy.Symbol(name)

    # Prefer calling by keyword, but only for signatures that actually accept it.
    # Falling back to positional arguments is reserved for a genuine
    # signature/arity mismatch: a ``TypeError`` raised *inside* the user callable
    # must propagate, not be masked by a second (positional) call.
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        try:
            signature.bind(**arguments)
        except TypeError:
            positional = list(variables.values()) + [sympy.Symbol(name) for name in names]
            signature.bind(*positional)
            return sympy.sympify(function(*positional))
        return sympy.sympify(function(**arguments))

    # No inspectable signature (e.g. some C callables): keep the previous
    # best-effort fallback.
    try:
        return sympy.sympify(function(**arguments))
    except TypeError:
        positional = list(variables.values()) + [sympy.Symbol(name) for name in names]
        return sympy.sympify(function(*positional))


def resolve(
    *,
    function: Callable | None = None,
    expression=None,
    sympy_expression: sympy.Expr | None = None,
    variables: Iterable[str] = ("x", "y", "z"),
    parameters: Mapping[str, float] | None = None,
    context: str = "field functor",
) -> tuple[sympy.Expr, str, Callable]:
    """
    Translate any of the expression/function/sympy spellings into all three.

    Parameters are kept symbolic in the returned expression; the canonical
    string and callable are the exact translations of it. If several spellings
    are supplied they must resolve to the same expression (a mismatch that still
    depends only on the coordinates is rejected).

    ``sympy_expression`` is the already-resolved spelling and is treated exactly
    like an ``expression`` (the two must agree if both are given), so a caller
    that only has the sympy form still gets the string and callable translations.

    Returns ``(symbolic_expression, canonical_string, callable)``. The returned
    callable has the named parameters already substituted (it depends on the
    coordinates only), matching the public ``<field>_function`` view; the
    symbolic expression keeps them as symbols so that the pypicongpu models can
    declare the parameters separately.
    """
    variables = tuple(variables)
    symbols = {name: sympy.Symbol(name) for name in variables}
    parameters = dict(parameters or {})
    # Parse the expression with the parameter names bound to symbols, so a
    # parameter whose name also exists in sympy's namespace (e.g. ``E1``) is
    # still parsed as that symbol rather than as the sympy object.
    parameter_locals = {**symbols, **{name: sympy.Symbol(name) for name in parameters}}

    spellings: dict[str, sympy.Expr] = {}
    if expression is not None:
        spellings["expression"] = (
            expression if isinstance(expression, sympy.Expr) else sympify_expression(expression, parameter_locals)
        )
    if sympy_expression is not None:
        spellings["sympy"] = (
            sympy_expression
            if isinstance(sympy_expression, sympy.Expr)
            else sympify_expression(sympy_expression, parameter_locals)
        )
    if function is not None:
        # The callable receives the parameters as (symbolic) keywords, exactly
        # like an expression string references them by name.
        spellings["function"] = expression_from_callable(function, symbols, parameters)

    if not spellings:
        raise ValueError(f"{context} must provide an expression, a sympy expression or a function.")

    # Every supplied spelling is compared with the parameters substituted, so a
    # symbolic expression and its (numeric) callable counterpart agree. This also
    # catches a string and a sympy spelling that disagree.
    #
    # ``sympy.simplify`` alone is not a reliable zero test for the floating-point
    # expressions that arise here: it cannot cancel ``cos(a) - cos(b)`` or reduce
    # a ``Piecewise`` difference when ``a`` and ``b`` denote the same value at
    # different sympy precisions (a common artifact of one spelling going through
    # the canonical decimal string and the other coming straight from the
    # callable).  Normalising every float to its 64-bit value first makes the
    # comparison robust without accepting genuinely different expressions.
    resolved = [_canonical_float64(value.subs(parameters)) for value in spellings.values()]
    reference = resolved[0]
    for name, value in zip(list(spellings)[1:], resolved[1:], strict=False):
        difference = sympy.simplify(reference - value)
        if difference != 0:
            raise ValueError(
                f"{context} {name} spelling disagrees by {difference!r}; provide one spelling, or several that agree."
            )

    symbolic = next(iter(spellings.values()))
    return symbolic, expression_string(symbolic), function_from_expression(symbolic.subs(parameters), variables)


class _FieldFunctor:
    """
    One sympy-backed field expression plus its named parameters.

    This is the reusable core shared by every field that is rendered into a
    generated C++ functor. It encapsulates the full pipeline once, so no caller
    has to reproduce it:

    1. resolve any of the interchangeable spellings (a PICMI string, a plain
       number, an already-parsed sympy expression or a callable of the
       coordinate variables) and check that they agree,
    2. normalise to a sympy expression and translate between the string, sympy
       and callable spellings,
    3. collect/validate the named parameters supplied as additional keyword
       arguments and reject undefined free symbols,
    4. render the expression through the :class:`PMAccPrinter`.

    Parameters
    ----------
    expression:
        A PICMI expression string, a plain number or an already-parsed sympy
        expression.
    function:
        A callable of the coordinate variables (in the order of ``variables``)
        returning something sympy can understand. Extra named parameters are
        taken from ``parameters``. May be combined with the other spellings if
        they agree; at least one spelling must be given.
    sympy_expression:
        An already-resolved :class:`sympy.Expr`; the exact ``*_sympy`` spelling.
        It is validated against any ``expression``/``function`` given and may be
        the only spelling supplied.
    variables:
        The names of the supported free variables, in argument order. The
        applied fields use ``("x", "y", "z", "t")``; the density and per-axis
        momentum/spread fields use ``("x", "y", "z")``.
    parameters:
        Mapping of parameter name to value (the PICMI ``user_defined_kw``).
        Values stay symbolic in :attr:`symbolic`; they are rendered as
        compile-time constants.
    context:
        Prefix used in error messages (e.g. ``"AnalyticAppliedField Ex"``).

    Attributes
    ----------
    symbolic:
        The resolved expression with parameters kept symbolic.
    sympy:
        :attr:`symbolic` with every parameter substituted by its value (the
        public ``<field>_sympy`` view).
    expression:
        The canonical PICMI string spelling (parameters symbolic).
    function:
        The callable spelling; substitutes the coordinate arguments into
        :attr:`sympy`.
    """

    def __init__(
        self,
        *,
        expression=None,
        function: Callable | None = None,
        sympy_expression: sympy.Expr | None = None,
        variables: Iterable[str] = ("x", "y", "z", "t"),
        parameters: Mapping[str, float] | None = None,
        context: str = "field functor",
    ):
        self.context = context
        self.variables = tuple(variables)
        self.parameters = dict(parameters or {})
        self.symbolic, self.expression, self.function = resolve(
            expression=expression,
            function=function,
            sympy_expression=sympy_expression,
            variables=self.variables,
            parameters=self.parameters,
            context=context,
        )
        self.sympy = self.symbolic.subs(self.parameters)

        check_parameter_names(self.parameters)
        self._check_symbols()

    def _check_symbols(self) -> None:
        allowed = set(self.variables) | set(self.parameters)
        check_allowed_symbols({"expression": self.symbolic}, allowed, self.context)

    def render(self) -> str:
        """
        The PMAcc C++ rendering, with parameters kept as named symbols.

        This is the form used by the pypicongpu ``BackgroundField``, which
        declares the parameters separately (see :meth:`parameter_list`).
        """
        return render(self.symbolic)

    def evaluate(self, *args):
        """
        Evaluate the field at the given coordinates (SI), with all parameters substituted.

        The arguments are the coordinate variables in the order of
        :attr:`variables` and may be scalars or numpy arrays (broadcasting is
        supported). Named parameters are already substituted, so no keyword
        arguments are accepted. Returns the numerical value(s), not a sympy
        expression; this is the same evaluation the pypicongpu layer renders
        into C++.
        """
        function = sympy.lambdify(tuple(sympy.Symbol(name) for name in self.variables), self.sympy, "numpy")
        values = function(*args)
        # lambdify returns a plain Python number for scalar, constant input;
        # wrap it into an array so the result type is consistent (and so a
        # caller can broadcast/compose it uniformly).
        return np.asarray(values)

    def parameter_list(self) -> list[dict]:
        """
        The named parameters as ``{"name": ..., "value": ...}`` dicts, sorted by name.

        The name is rendered through the :class:`PMAccPrinter`, so it is the
        spelling that actually appears in the generated functor (escaped
        keywords included). Re-rendering it is idempotent, so passing it through
        the pypicongpu model validators a second time is safe.
        """
        return [{"name": render_identifier(name), "value": value} for name, value in sorted(self.parameters.items())]
