"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import logging
from itertools import chain
from functools import partial, wraps
from inspect import Parameter, signature
from operator import itemgetter
from typing import Any, Self


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
    Constructs a function that can be called with the first parameter as positional argument regardless of it being kw-only or not.
    """
    if parameter.kind in [Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD]:
        return lambda d: f(**{parameter.name: d}, **kwargs)
    elif parameter.kind in [Parameter.POSITIONAL_ONLY]:
        return lambda d: f(d, **kwargs)
    else:
        # The remaining option for parameter.kind is VAR_KEYWORD (at the time of writing)
        # and that was caught above already.
        raise Exception("This path should be unreachable!")


def decorating_class(cls_or_name, parameter=None):
    """
    A decorating class can be used as decorator, i.e., in the following example `a` and `b` are identical:

        a = MyClass(b=lambda: print("Hello World!"), c=6)

        @MyClass(c=6)
        def b():
            print("Hello World!")
    """
    if isinstance(cls_or_name, str):
        return lambda cls: decorating_class(cls, parameter=Parameter(name=cls_or_name, kind=Parameter.KEYWORD_ONLY))
    # It is important to extract the signature before decorating the class.
    # Otherwise, we'll only see the names of the decorator's arguments.
    parameter = parameter or _extract_first_parameter(cls_or_name)

    @wraps(cls_or_name, updated=tuple())
    class Tmp(cls_or_name):
        def __new__(cls, decorated=None, **kwargs):
            decorated = kwargs.pop(parameter.name, None) or decorated
            if decorated is None:
                return _pass_first_parameter_to(cls, parameter, kwargs)
            constructor = partial(super().__new__, cls)
            try:
                return _pass_first_parameter_to(constructor, parameter, kwargs)(decorated)
            except TypeError:
                return constructor()

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


class _Attribute(str):
    pass


class _Item:
    args: Any

    def __init__(self, args):
        self.args = args


class UnpackChain:
    """
    Helper class to iterate over (nested) members.

    This class can wrap another object to allow
    iterating over recursively unpacked members.
    This is very helpful to iterate over
    deeply nested objects as they easily occur with pydantic.

    By example (using a slightly artifical recursive model):

        from pydantic import BaseModel

        class Model(BaseModel):
            a: list["Model"] | int
            b: list[list["Model"]] = Field(default_factory=list)

        m = Model(a=[{"a": [{"a": 1}, {"a": 2}], "b": [[{"a": 4}, {"a": 5}]]}, {"a": 3}])
        print(*(x for x in UnpackChain(m).a.a))
        #> [Model(a=1, b=[]), Model(a=2, b=[])] 3
        print(*(x for x in UnpackChain(m).a.a.a))
        #> 1 2
        print(*(x for x in UnpackChain(m).a.b[:].a))
        #> 4 5
        print(*(x for x in UnpackChain(m).a.b[0].a))
        #> 4

    Warning: Consecutive access by []-operator is ambiguous and tricky!
    Only the last in a series of []-accesses is allowed to
    generate something non-iterable. So, the following are all fine:
        - [:]
        - [0]
        - [:][:]
        - [:][0]
        - [0].a[:]
    But the following fail:
        - [0][:]
        - [0][0]
    This is because I cannot tell purely from the arguments,
    if an access by [] is supposed to imply iteration or not.
    The semantics are currently that it DOES imply iteration
    (except for the last of a series of [] which is a bit more relaxed).
    """

    def __init__(self, obj, requests=None):
        self._requests = requests or []
        self.obj = obj

    def __getattr__(self, name: str, /) -> Self:
        self._requests.append(_Attribute(name))
        return self

    def __getitem__(self, *args):
        self._requests.append(_Item(args))
        return self

    def values(self):
        self._requests.append("values()")
        return self

    def __iter__(self):
        if len(self._requests) == 0:
            return iter([self.obj])

        new_obj = (
            alt(
                # Using itemgetter here because indexing via [*args] apparently doesn't work?
                lambda: itemgetter(*self._requests[0].args)(self.obj),
                lambda: getattr(self.obj, self._requests[0]),
                NotImplemented,
            )
            if self._requests[0] != "values()"
            else alt(lambda: self.obj.values(), NotImplemented)
        )

        if new_obj is NotImplemented:
            return iter([])

        if len(self._requests) == 1 or alt(lambda: hasattr(new_obj, self._requests[1]), False):
            return iter(UnpackChain(new_obj, requests=self._requests[1:]))

        return chain(*(UnpackChain(x, requests=self._requests[1:]) for x in alt(lambda: iter(new_obj), [])))


def unique(iterable):
    # very naive, just for non-hashables that can still be compared
    result = []
    for x in iterable:
        if x not in result:
            result.append(x)
    return result


def unsupported(name: str, value: Any = 1, default: Any = None) -> None:
    """
    Print a msg that the feature/parameter/thing is unsupported.

    If 2nd param (value) and 3rd param (default) are set:
    supress msg if value == default

    If 2nd param (value) is set and 3rd is missing:
    supress msg if value is None

    If only 1st param (name) is set: always print msg

    :param name: name of the feature/parameter/thing that is unsupported
    :param value: If set: only print warning if this is not None
    :param default: If set: check value against this param instead of none
    """

    if value != default:
        logging.warning("unsupported: {}".format(name))
