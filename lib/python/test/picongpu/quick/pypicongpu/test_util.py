"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from dataclasses import dataclass
from typing import Annotated, Callable
from pydantic import BaseModel, ValidationError
from unittest import TestCase
from picongpu.pypicongpu.util import (
    UnsupportedFeatureError,
    UnpackChain,
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


class TestUnpackChain(TestCase):
    def test_zero_length_chain(self):
        obj = [1, 2, 3]
        self.assertListEqual(obj, list(UnpackChain(obj)))

    def test_indexing_with_index(self):
        obj = [1, 2, 3]
        self.assertListEqual([obj[0]], list(UnpackChain(obj)[0]))

    def test_indexing_with_slice(self):
        obj = [1, 2, 3]
        self.assertListEqual(obj[:], list(UnpackChain(obj)[:]))

    def test_simple_attribute_access(self):
        class Obj:
            my_tuple = (1, 2, 3)

        self.assertListEqual(list(Obj().my_tuple), list(UnpackChain(Obj).my_tuple))

    def test_nested_attribute_access(self):
        class InternalObj:
            another_tuple = (4, 5)

        class Obj:
            my_tuple = (InternalObj(), InternalObj())

        self.assertListEqual(2 * [*InternalObj().another_tuple], list(UnpackChain(Obj).my_tuple.another_tuple))

    def test_nested_attribute_access_to_toplevel(self):
        # We need a comparison operator for this one:
        @dataclass
        class InternalObj:
            another_tuple = (4, 5)

        class Obj:
            my_tuple = (InternalObj(), InternalObj())

        self.assertListEqual(2 * [InternalObj()], list(UnpackChain(Obj).my_tuple))

    def test_method_calls(self):
        obj = {"a": 1, "b": 2}
        self.assertListEqual(list(obj.values()), list(UnpackChain(obj).values()))
        self.assertListEqual(list(obj.keys()), list(UnpackChain(obj).keys()))
        self.assertListEqual(list(obj.items()), list(UnpackChain(obj).items()))

    def test_deeply_nested_complex_object(self):
        class InternalObj1:
            my_dict = {"a": 1, "b": 2}

        class InternalObj2:
            another_tuple = (InternalObj1(), InternalObj1())

        class CustomDict:
            def values(self):
                return [1, 2]

        class InternalObj3:
            my_dict = CustomDict()

        class InternalObj4:
            another_tuple = (InternalObj1(), InternalObj3())

        class Obj:
            my_tuple = (InternalObj1(), InternalObj2(), InternalObj4())

        self.assertListEqual(
            4 * [*InternalObj1().my_dict.values()], list(UnpackChain(Obj).my_tuple.another_tuple.my_dict.values())
        )
        self.assertListEqual([*InternalObj1().my_dict.values()], list(UnpackChain(Obj).my_tuple.my_dict.values()))

    def test_consecutive_indexing(self):
        obj = [[[1.2], [3, 4]], [5, [[6], [7]]]]
        self.assertListEqual(obj, list(UnpackChain(obj)))
        self.assertListEqual(obj[0], list(UnpackChain(obj)[0]))
        self.assertListEqual(obj[:], list(UnpackChain(obj)[:]))
        with self.assertRaises(ValueError):
            self.assertListEqual(obj[0][:], list(UnpackChain(obj)[0][:]))

    def test_with_pydantic(self):
        from pydantic import BaseModel, Field

        class Model(BaseModel):
            a: list["Model"] | int
            b: list[list["Model"]] = Field(default_factory=list)

        m = Model(a=[{"a": [{"a": 1}, {"a": 2}], "b": [[{"a": 4}, {"a": 5}]]}, {"a": 3}])

        self.assertListEqual([Model(a=1, b=[]), Model(a=2, b=[]), 3], list(UnpackChain(m).a.a))
        self.assertListEqual([1, 2], list(UnpackChain(m).a.a.a))
        self.assertListEqual([4, 5], list(UnpackChain(m).a.b[0].a))
        self.assertListEqual([], list(UnpackChain(m).a.b.a))
        # It is unfortunate that this doesn't produce [4, 5], but it's logically sound:
        self.assertListEqual([], list(UnpackChain(m).a.b[:].a))
