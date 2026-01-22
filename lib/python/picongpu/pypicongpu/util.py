"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import typeguard
import typing
import logging

attr_cnt = 0


def alt(expr, alternative, *exprs, ignore=(AttributeError, TypeError, IndexError)):
    """Try to evaluate the expression and return the first valid.

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
        assert alt(lambda: t['data'], d).count(1) == 2

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


def unique(iterable):
    # very naive, just for non-hashables that can still be compared
    result = []
    for x in iterable:
        if x not in result:
            result.append(x)
    return result


# note: type_ may be either a type, or a definition by typing
# depending on the python version the type of typing.XXXX is different
# (_GenericMeta vs. GenericMeta) -- so we compute it on the fly
@typeguard.typechecked
def build_typesafe_property(
    type_: typing.Union[type, type(typing.List[int])], name: typing.Optional[str] = None
) -> property:
    if name is None:
        global attr_cnt
        name = str(attr_cnt)
        attr_cnt += 1
    # don't use private prefix '__' to avoid name mangling
    actual_var_name = "magic_string_private_____{}".format(name)

    @typeguard.typechecked
    def getter(self) -> type_:
        if not hasattr(self, actual_var_name):
            raise AttributeError("variable is not initialized")
        return getattr(self, actual_var_name)

    @typeguard.typechecked
    def setter(self, value: type_):
        setattr(self, actual_var_name, value)

    return property(getter, setter)


@typeguard.typechecked
def unsupported(name: str, value: typing.Any = 1, default: typing.Any = None) -> None:
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
