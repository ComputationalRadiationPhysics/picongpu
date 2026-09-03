"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Annotated, Callable
from pydantic import BaseModel, ValidationError
from unittest import TestCase
from picongpu.pypicongpu.util import (
    UnsupportedFeatureError,
    decorating_class,
    rejects_unsupported,
    unsupported,
)


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


class TestUnsupported(TestCase):
    def test_unsupported_only_name_always_raises(self):
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            unsupported("some feature")
        self.assertIn("some feature", str(ctx.exception))

    def test_unsupported_value_equal_default_is_allowed(self):
        # must not raise
        unsupported("some feature", value=None, default=None)
        unsupported("some feature", value=42, default=42)

    def test_unsupported_value_different_from_default_raises(self):
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            unsupported("some feature", value=41, default=42)
        self.assertIn("some feature", str(ctx.exception))
        self.assertIn("41", str(ctx.exception))

    def test_unsupported_none_value_raises_unless_default_is_none(self):
        with self.assertRaises(UnsupportedFeatureError):
            unsupported("some feature", value=None, default=1)
        # must not raise
        unsupported("some feature", value=None, default=None)

    def test_unsupported_list_tuple_mixup_is_compared_elementwise(self):
        # must not raise: same content, different container types
        unsupported("some feature", value=[1, 2], default=(1, 2))
        with self.assertRaises(UnsupportedFeatureError):
            unsupported("some feature", value=[1, 2], default=(1, 3))

    def test_error_attributes(self):
        try:
            unsupported("feature x", value=7)
        except UnsupportedFeatureError as error:
            self.assertEqual(error.feature, "feature x")
            self.assertEqual(error.given, 7)
        else:
            self.fail("expected UnsupportedFeatureError")

    def test_rejects_unsupported_at_construction(self):
        class Model(BaseModel):
            stencil_order: Annotated[int | None, rejects_unsupported("higher order solvers")] = None

        # default and explicit None must be accepted
        self.assertIsNone(Model().stencil_order)
        self.assertIsNone(Model(stencil_order=None).stencil_order)
        # a real value must be rejected, surfacing as a ValidationError naming the feature
        with self.assertRaises(ValidationError) as ctx:
            Model(stencil_order=4)
        self.assertIn("higher order solvers", str(ctx.exception))
        self.assertIn("4", str(ctx.exception))

    def test_rejects_unsupported_custom_default(self):
        class Model(BaseModel):
            n_pass: Annotated[int, rejects_unsupported("smoother n_pass", default=2)] = 2

        self.assertEqual(Model().n_pass, 2)
        self.assertEqual(Model(n_pass=2).n_pass, 2)
        with self.assertRaises(ValidationError):
            Model(n_pass=4)
