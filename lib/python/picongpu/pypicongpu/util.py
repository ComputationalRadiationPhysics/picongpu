"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from itertools import chain
from functools import wraps
from inspect import Parameter, signature
from operator import itemgetter, attrgetter, methodcaller
from typing import Any

from pydantic import BeforeValidator


def _extract_first_parameter(cls):
    sig = signature(cls).parameters
    try:
        parameter = next(iter(sig.values()))
    except StopIteration as error:
        raise TypeError(
            f"A decorating class must have at least one argument to its constructor. You gave: {sig=} for {cls=}."
        ) from error
    if parameter.kind in [Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL]:
        raise TypeError(
            f"A decorating class cannot have variadic (keyword) arguments to its constructor first. You gave: {sig=} for {cls=}."
        )
    return parameter


def _pass_first_parameter_to(f, parameter, kwargs):
    """
    Constructs a callable that passes its argument as the first constructor
    parameter, regardless of that parameter being keyword-only or positional-only.
    """
    if parameter.kind is Parameter.POSITIONAL_ONLY:
        return lambda decorated: f(decorated, **kwargs)
    return lambda decorated: f(**{parameter.name: decorated}, **kwargs)


def decorating_class(cls_or_name, parameter=None):
    """
    A decorating class can be used as decorator, i.e., in the following example `a` and `b` are identical:

        a = MyClass(b=lambda: print("Hello World!"), c=6)

        @MyClass(c=6)
        def b():
            print("Hello World!")

    Instead of a class, a string can be given which is used as the name of the
    constructor parameter that receives the decorated object:

        @decorating_class("density_function")
        class AnalyticDistribution(...):
    """
    if isinstance(cls_or_name, str):
        name = cls_or_name
        return lambda cls: decorating_class(cls, parameter=Parameter(name=name, kind=Parameter.KEYWORD_ONLY))
    # It is important to extract the signature before decorating the class.
    # Otherwise, we'll only see the names of the decorator's arguments.
    parameter = parameter or _extract_first_parameter(cls_or_name)

    @wraps(cls_or_name, updated=tuple())
    class Tmp(cls_or_name):
        def __init__(self, decorated=None, **kwargs):
            """Turn a positional decorator argument into the named keyword.

            ``type.__call__`` forwards the *original* call arguments to ``__init__``
            after ``__new__`` returns. Pydantic ``BaseModel.__init__`` only accepts
            keyword arguments, so a leading positional argument is re-passed as the
            named keyword here.
            """
            if decorated is not None:
                if parameter.name in kwargs:
                    raise TypeError(f"got multiple values for argument '{parameter.name}'")
                super().__init__(**{parameter.name: decorated}, **kwargs)
            else:
                super().__init__(**kwargs)

        def __new__(cls, decorated=None, **kwargs):
            if decorated is None and parameter.name in kwargs:
                decorated = kwargs.pop(parameter.name)
            if decorated is None:
                # @MyClass(extra=...) -- no decorated object (yet):
                # return a callable that accepts the future @decorator.
                return _pass_first_parameter_to(cls, parameter, kwargs)
            # @MyClass(object) -- build an instance. Call the original __new__ if it
            # accepts the decorated argument (classes with a custom __new__); for
            # plain classes and pydantic models this raises, so fall back to a bare
            # instance whose __init__ does the actual construction.
            try:
                return super().__new__(cls, **{parameter.name: decorated}, **kwargs)
            except TypeError:
                return object.__new__(cls)

    return Tmp


def alt(expr, alternative, *exprs, ignore=(AttributeError, TypeError, IndexError)):
    """
    Try to evaluate the expression and return the first valid.

    This basically allows for runtime SFINAE ("substitution failure is not an error")
    as with the default `ignore` argument it does not raise an error
    if the substitution of an expression failed.
    Expressed differently, this enables local polymorphism
    by listing different alternatives to reach
    the "same" (or a reasonable-in-this-context) answer
    from a number of different objects.
    For example:

        # This is a dictionary:
        d = {'data': (1,2,1)}
        assert alt(lambda: d['data'], d).count(1) == 2

        # This is a tuple:
        d = d['data']
        assert alt(lambda: d['data'], d).count(1) == 2

    This is very helpful when looping over heterogeneous lists:

        # `d` is the dictionary from above:
        my_list = [1, 'string', d]
        keys = [keys for el in my_list for keys in alt(lambda: el.keys(), [])]

    If multiple expressions are given,
    this will try to evaluate one after another
    until any of them succeeds or the last alternative is returned.
    So, more precisely the signature is `(expr, *exprs, alternative)`
    but writing it like this would make `alternative` kw-only.
    The alternative can itself be an expression.
    """
    errors = []
    try:
        return expr()
    except ignore as error1:
        errors.append(error1)
        if len(exprs) > 0:
            return alt(alternative, *exprs, ignore=ignore)
        try:
            return alternative()
        except TypeError as error2:
            errors.append(error2)
            return alternative


class UnpackChain:
    """
    Helper class to iterate over (nested) members.

    This class can wrap another object to allow
    iterating over recursively unpacked members.
    This is very helpful to iterate over
    deeply nested objects as they easily occur with pydantic.

    Warning: Consecutive access by []-operator is ambiguous and therefore forbidden.
    """

    def __init__(self, obj, requests=None):
        self._requests = requests or tuple()
        self.obj = obj

    def __getattr__(self, name: str, /):
        return UnpackChain(self.obj, requests=self._requests + (attrgetter(name),))

    def __getitem__(self, *args):
        if alt(lambda: isinstance(self._requests[-1], itemgetter), False):
            raise ValueError("Consecutive access by []-operator is ambiguous and forbidden.")
        return UnpackChain(self.obj, requests=self._requests + (itemgetter(*args),))

    def __call__(self, *args, **kwargs):
        return UnpackChain(self.obj, requests=self._requests + (methodcaller("__call__", *args, **kwargs),))

    def values(self):
        return UnpackChain(self.obj, requests=self._requests + (methodcaller("values"),))

    def __iter__(self):
        if len(self._requests) == 0:
            try:
                return iter(self.obj)
            except TypeError:
                return iter([self.obj])

        new_obj = alt(lambda: self._requests[0](self.obj), NotImplemented)
        if new_obj is NotImplemented:
            return iter([])

        if len(self._requests) == 1 or alt(lambda: self._requests[1](new_obj), False):
            return iter(UnpackChain(new_obj, requests=self._requests[1:]))

        return chain(*(UnpackChain(x, requests=self._requests[1:]) for x in alt(lambda: iter(new_obj), [])))


def unique(iterable):
    # very naive, just for non-hashables that can still be compared
    result = []
    for x in iterable:
        if x not in result:
            result.append(x)
    return result


class UnsupportedFeatureError(ValueError):
    """
    Raised when the user requests a feature/parameter that PIConGPU does not implement.

    Unifies the error for both rejection points:
      - at construction time, via `rejects_unsupported` (pydantic models), and
      - at PICMI-to-pypicongpu conversion time, via `unsupported` (call sites that
        can only know the offending value after translation).
    """

    def __init__(self, feature: str, given: Any):
        self.feature = feature
        self.given = given
        super().__init__(
            f"'{feature}' is not (yet) implemented by PIConGPU. "
            f"You gave: {given!r}. Leave the parameter at its supported value, "
            "or track the feature at https://github.com/ComputationalRadiationPhysics/picongpu (label: PICMI)."
        )


def _handle_unsupported(feature: str, given: Any) -> None:
    """
    Single choke point for reacting to unsupported features.

    Currently always raises. Centralized so that the behaviour
    (raise/warn/ignore) can be made configurable in one place later on.
    """

    raise UnsupportedFeatureError(feature, given)


def _normalize_for_comparison(value: Any) -> Any:
    """List and tuple are compared element-wise, so a list/tuple mix-up isn't a mismatch."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def rejects_unsupported(feature: str, *, default: Any = None) -> BeforeValidator:
    """
    Return a pydantic BeforeValidator for a PICMI-standard field that PIConGPU
    does not implement yet.

    The field keeps its standard type (and usual default), so construction stays
    standard-conformant; any value different from `default` (the accepted no-op
    value, usually the standard default) raises `UnsupportedFeatureError`,
    which pydantic surfaces as a `ValidationError` naming the feature.

    Usage:
        class Model(BaseModel):
            stencil_order: Annotated[int | None, rejects_unsupported("higher order solvers")] = None
    """

    def _check(value: Any) -> Any:
        if _normalize_for_comparison(value) != _normalize_for_comparison(default):
            _handle_unsupported(feature, value)
        return value

    return BeforeValidator(_check)


_always_unsupported = object()


def unsupported(name: str, value: Any = _always_unsupported, default: Any = None) -> None:
    """
    Raise `UnsupportedFeatureError` for the feature/parameter `name`.

    If 2nd param (value) and 3rd param (default) are set:
    raise if value != default

    If 2nd param (value) is set and 3rd is missing:
    raise if value is not None

    If only 1st param (name) is set: always raise

    :param name: name of the feature/parameter/thing that is unsupported
    :param value: If set: only raise if this is not None
    :param default: If set: check value against this param instead of none
    """

    if value is _always_unsupported or _normalize_for_comparison(value) != _normalize_for_comparison(default):
        _handle_unsupported(name, None if value is _always_unsupported else value)
