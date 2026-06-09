# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Callable
from pydantic import BaseModel
from unittest import TestCase
from picongpu.pypicongpu.util import decorating_class


@decorating_class
class Functor:
    def __init__(self, functor, **kwargs):
        self.functor = functor
        self.kwargs = kwargs


class TestDecoratingClass(TestCase):
    def assert_functors_equal(self, result, expected):
        self.assertDictEqual(result.kwargs, expected.kwargs)
        # some simple probing if the functions are identical
        for i in range(10):
            self.assertEqual(result.functor(i), expected.functor(i))

    def test_simple_functor(self):
        @Functor
        def result(x):
            return x

        expected = Functor(lambda x: x)
        self.assert_functors_equal(result, expected)

    def test_simple_functor_with_kwargs(self):
        kwargs = {"a": 1}

        @Functor(**kwargs)
        def result(x):
            return x

        expected = Functor(lambda x: x, **kwargs)
        self.assert_functors_equal(result, expected)
        self.assertDictEqual(result.kwargs, kwargs)

    def test_simple_functor_with_duplicate_arg_raises(self):
        kwargs = {"functor": lambda x: x}
        with self.assertRaises(TypeError):
            Functor(kwargs["functor"], **kwargs)

    def test_no_args_forbidden(self):
        with self.assertRaises(TypeError):

            @decorating_class
            class _:
                def __init__(self):
                    pass

    def test_variadic_args_forbidden(self):
        with self.assertRaises(TypeError):

            @decorating_class
            class _:
                def __init__(self, *_):
                    pass

    def test_variadic_kwargs_only_forbidden(self):
        with self.assertRaises(TypeError):

            @decorating_class
            class _:
                def __init__(self, **_):
                    pass

    def test_with_pydantic(self):
        @decorating_class
        class PydanticFunctor(BaseModel):
            functor: Callable[[int], int]
            # The naming here is chosen such that self.assert_functors_equal still works.
            kwargs: dict[str, int]

        remaining_args = (2,)
        kwargs = {"a": 1}

        @PydanticFunctor(remaining_args=remaining_args, kwargs=kwargs)
        def result(x):
            return x

        expected = PydanticFunctor(functor=lambda x: x, remaining_args=remaining_args, kwargs=kwargs)
        self.assert_functors_equal(result, expected)
        self.assertDictEqual(result.kwargs, kwargs)

    def test_forwarding_arguments_to_new(self):
        given_kwargs = {"a": 1}

        @decorating_class
        class SpecialFunctor:
            def __new__(cls, functor, **kwargs):
                assert kwargs == given_kwargs
                return super().__new__(cls)

            def __init__(self, functor, **kwargs):
                self.functor = functor
                self.kwargs = kwargs

        @SpecialFunctor(**given_kwargs)
        def result(x):
            return x

        expected = SpecialFunctor(lambda x: x, **given_kwargs)
        self.assert_functors_equal(result, expected)
        self.assertDictEqual(result.kwargs, given_kwargs)
