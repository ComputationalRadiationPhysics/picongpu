# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import inspect
from typing import Callable

from pydantic import BaseModel, ValidationError


def has_attribute(instance, name):
    if isinstance(instance, type) and issubclass(instance, BaseModel):
        return name in instance.model_fields or name in map(lambda x: x.alias, instance.model_fields.values())

    # It should be this:
    #
    #     return hasattr(instance, name)
    #
    # But this seems to interact weirdly with our util.build_typesafe_property.
    #
    # This version works fine but throws a warning for pydantic models.
    # We could get rid of this by using instance.model_dump() instead.
    return name in dir(instance)


def _sanitize_conversions(conversions, from_instance, to_instance):
    conversions = conversions or {}

    for new_key, old in conversions.items():
        if not has_attribute(to_instance, new_key):
            failed_conversion = {new_key: old}
            message = (
                f"Conversion failed. '{new_key}' is not a member of the receiver {to_instance}. "
                f"You gave {failed_conversion}."
            )
            raise ValueError(message)
        if isinstance(old, str) and not has_attribute(from_instance, old):
            failed_conversion = {new_key: old}
            message = (
                f"Conversion failed. '{old}' is not a member of the provider {from_instance}. "
                f"You gave {failed_conversion}."
            )
            raise ValueError(message)

    return {name: _value_generator(old) if isinstance(old, str) else old for name, old in conversions.items()}


def _value_generator(name):
    return lambda self: getattr(self, name)


def copy_attributes(
    from_instance,
    to,
    conversions: None | dict[str, str | Callable] = None,
    remove_prefix: str = "",
    ignore=tuple(),
    default_converter=lambda self: self,
):
    """
    Copy attributes from one object to another.

    This function copies attributes from the `from_instance` to `to` if `to` is a class instance.
    If `to` is a class, it will try to construct and return an instance filled with values from `from_instance`.

    This function only copies attributes under the following circumstances:
        - The attribute exists in `to`.
        - The attribute is not private in `from_instance`.

    Optionally, `conversions` allows to define custom mappings between attributes.
    Keys are strings and interpreted as the name of the attribute to be set on `to`.
    For the values, two semantics are supported:
        - If a `value` is a string, the `(key, value)` pair is interpreted as renaming, i.e.
          `to.key = from_instance.value`.
        - Otherwise, `value` must be a Callable that takes `from_instance` as first and only argument
          and returns the value that `key` is supposed to have in `to`, i.e.
          `to.key = value(from_instance)`.

    Further useful features:
        - `remove_prefix` allows to remove a prefix from `from_instance` member names
          before looking them up and inserting them into `to`.
        - `ignore` allows to ignore some attributes in `from_instance` and not copy them.
          A custom conversion takes precedence and overrides this behaviour.
        - `default_converter` is applied to all values retrieved from `from_instance`
          before they are put into `to`.
    """
    assignments = {
        to_name: _value_generator(from_name)
        for from_name, _ in (
            type(from_instance).model_fields.items()
            if isinstance(from_instance, BaseModel)
            else inspect.getmembers(from_instance)
        )
        if from_name not in ignore
        and not from_name.startswith("_")
        and has_attribute(to, to_name := from_name.removeprefix(remove_prefix))
    } | _sanitize_conversions(conversions, from_instance, to)

    # This is a two-pass process because after generating the defaults
    # we had to apply the custom conversions on top.
    assignments = {
        key: default_converter(value_generator(from_instance)) for key, value_generator in assignments.items()
    }

    if isinstance(to, type):
        try:
            # First case: `to` is a class and can be constructed with a fully-qualified constructor call (pydantic.BaseModel).
            return to(**assignments)
        except TypeError:
            try:
                # Second case: `to` is a default-constructible class to which we can copy attributes afterwards.
                to_instance = to()
            except (ValidationError, TypeError) as e:
                message = (
                    "Instantiation failed. The receiving class must be default constructible. "
                    f"You gave {to} which expects {len(inspect.signature(to.__init__).parameters) - 1} argument in its constructor. "
                    "You can work with an instance instead of a class in this case."
                )
                raise ValueError(message) from e
            # We've got an instance now, proceed via the path for instances.
            return copy_attributes(
                from_instance,
                to_instance,
                conversions=conversions,
                remove_prefix=remove_prefix,
                ignore=ignore,
                default_converter=default_converter,
            )
    else:
        # Third case: We've been given an instance directly. Copy over attributes.
        for key, value in assignments.items():
            setattr(to, key, value)
        return to


def converts_to(
    to_class,
    conversions=None,
    preamble=None,
    remove_prefix="",
    ignore=tuple(),
    default_converter=lambda self, *args, **kwargs: self,
):
    """
    Add a get_as_pypicongpu method that uses copy_attributes.

    This class decorator adds a method `get_as_pypicongpu`
    that copies the content of `self` to a new instance of `to_class`
    by virtue of the `copy_attributes` function.
    See the `copy_attributes` docstring for details.

    If given, it runs `preamble(self)` before copying.
    """
    if has_attribute(to_class, "get_as_pypicongpu"):
        message = f"Adding 'get_as_pypicongpu' failed because it already existed on receiving class {to_class}."
        raise TypeError(message)

    def decorator(cls):
        def get_as_pypicongpu(self, *args, **kwargs):
            if preamble:
                preamble(self, *args, **kwargs)
            local_conversions = {
                key: value
                if isinstance(value, str)
                # Python binds lambdas by reference, so if we were to use the seemingly obvious implementation,
                # all lambdas in this dictionary would use the last value of `value`.
                # That is why we need to force immediate evaluation:
                else (lambda local_value: lambda self: local_value(self, *args, **kwargs))(value)
                for key, value in (conversions or {}).items()
            }
            return copy_attributes(
                self,
                to_class,
                conversions=local_conversions,
                remove_prefix=remove_prefix,
                ignore=ignore,
                default_converter=lambda self: default_converter(self, *args, **kwargs),
            )

        cls.get_as_pypicongpu = get_as_pypicongpu
        return cls

    return decorator


def default_converts_to(to_class, conversions=None, preamble=None, remove_prefix="", ignore=tuple()):
    return converts_to(
        to_class,
        conversions=conversions,
        preamble=preamble
        or (lambda self, *args, **kwargs: self.check(*args, **kwargs) if has_attribute(self, "check") else None),
        remove_prefix=remove_prefix or "picongpu_",
        ignore=ignore or ("check",),
        default_converter=lambda self, *args, **kwargs: self.get_as_pypicongpu(*args, **kwargs)
        if has_attribute(self, "get_as_pypicongpu")
        else self,
    )
