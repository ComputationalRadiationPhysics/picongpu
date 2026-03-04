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
        self.remaining_args = tuple()
        self.kwargs = kwargs


class TestDecoratingClass(TestCase):
    def assert_functors_equal(self, result, expected):
        self.assertDictEqual(result.kwargs, expected.kwargs)
        self.assertTupleEqual(result.remaining_args, expected.remaining_args)
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
        self.assertTupleEqual(result.remaining_args, tuple())
        self.assertDictEqual(result.kwargs, kwargs)

    def test_simple_functor_with_duplicate_arg_raises(self):
        kwargs = {"functor": lambda x: x}
        with self.assertRaises(TypeError):
            Functor(kwargs["functor"], **kwargs)

    def test_no_args(self):
        with self.assertRaises(TypeError):

            @decorating_class
            class NoArgs:
                def __init__(self):
                    pass

    def test_parameter_packs(self):
        @decorating_class
        class VariadicFunctor:
            def __init__(self, *args, **kwargs):
                self.functor = args[0]
                self.remaining_args = args[1:]
                self.kwargs = kwargs

        @VariadicFunctor
        def result(x):
            return x

        expected = VariadicFunctor(lambda x: x)
        self.assert_functors_equal(result, expected)

    def test_variadic_kwargs_only(self):
        with self.assertRaises(TypeError):

            @decorating_class
            class VariadicFunctor:
                def __init__(self, **kwargs):
                    self.functor = kwargs.pop("functor", None)
                    self.remaining_args = tuple()
                    self.kwargs = kwargs

    def test_with_pydantic(self):
        @decorating_class
        class PydanticFunctor(BaseModel):
            functor: Callable[[int], int]
            # The naming here is chosen such that self.assert_functors_equal still works.
            remaining_args: tuple[int]
            kwargs: dict[str, int]

        remaining_args = (2,)
        kwargs = {"a": 1}

        @PydanticFunctor(remaining_args=remaining_args, kwargs=kwargs)
        def result(x):
            return x

        expected = PydanticFunctor(functor=lambda x: x, remaining_args=remaining_args, kwargs=kwargs)
        self.assert_functors_equal(result, expected)
        self.assertTupleEqual(result.remaining_args, remaining_args)
        self.assertDictEqual(result.kwargs, kwargs)
