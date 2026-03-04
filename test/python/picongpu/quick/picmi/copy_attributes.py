"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase
import inspect
import typeguard
import pytest
from picongpu.picmi.copy_attributes import copy_attributes, converts_to, default_converts_to

CLASS_NAME = "TmpClass"
ARBITRARY_VALUE = 42


def custom_conversion(self):
    return 2 * self.arbitrary_name


def gen_class(attributes=None, use_values=True):
    attributes = {key: value if use_values else None for key, value in (attributes or {}).items()}
    return type(CLASS_NAME, tuple(), attributes)


def gen_two_classes(common_attributes=None, only_provider=None, only_receiver=None):
    return gen_class((common_attributes or {}) | (only_provider or {})), gen_class(
        use_values=False, attributes=(common_attributes or {}) | (only_receiver or {})
    )


def gen_provider_and_matching_receiver_class(common_attributes=None, only_provider=None, only_receiver=None):
    classes = gen_two_classes(
        common_attributes=common_attributes or {}, only_provider=only_provider or {}, only_receiver=only_receiver or {}
    )
    return classes[0](), classes[1]


class MiniMock:
    counter = 0
    args = None
    kwargs = None

    def __call__(self, *args, **kwargs):
        self.counter += 1
        self.args = args
        self.kwargs = kwargs


@typeguard.typechecked
class ClassWithProperty:
    _attribute = 0

    @property
    def attribute(self):
        return self._attribute


class TestCopyAttributes(TestCase):
    def test_returns_instance_of_correct_class(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class()
        assert isinstance(copy_attributes(dummy_provider, DummyReceiver), DummyReceiver)

    def test_receiving_classes_must_be_default_constructible(self):
        # This does not apply to instances!

        def custom_init(self, _):
            # Setting this as __init__ makes the class expect one argument upon instantiation.
            # It is thus no longer default constructible.
            pass

        dummy_provider = gen_class()()
        DummyReceiver = gen_class(attributes={"__init__": custom_init})

        with pytest.raises(
            ValueError,
            match="Instantiation failed. The receiving class must be default constructible. "
            f".*{CLASS_NAME}.* which expects 1 argument in its constructor. "
            "You can work with an instance instead of a class in this case.",
        ):
            # This fails because it tries to instantiate DummyReceiver as DummyReceiver().
            copy_attributes(dummy_provider, DummyReceiver)
            # This would have worked:
            # copy_attributes(dummy_provider, DummyReceiver(1))

    def test_returns_identical_instance(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class()
        dummy_receiver = DummyReceiver()
        assert copy_attributes(dummy_provider, dummy_receiver) is dummy_receiver

    def test_copies_single_attribute(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class(
            common_attributes={"arbitrary_name": ARBITRARY_VALUE}
        )

        # precondition, just to check that our test infrastructure works:
        assert dummy_provider.arbitrary_name == ARBITRARY_VALUE
        assert dummy_provider.arbitrary_name != DummyReceiver().arbitrary_name

        assert copy_attributes(dummy_provider, DummyReceiver).arbitrary_name == ARBITRARY_VALUE

    def test_does_not_copy_private_attributes(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class(
            common_attributes={"arbitrary_name": ARBITRARY_VALUE, "_private_attribute": ARBITRARY_VALUE}
        )

        assert (
            copy_attributes(dummy_provider, DummyReceiver)._private_attribute
            != dummy_provider._private_attribute
        )

    def test_does_create_new_attributes(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class(
            common_attributes={"arbitrary_name": ARBITRARY_VALUE},
            only_provider={"additional_argument": ARBITRARY_VALUE},
        )

        assert not hasattr(copy_attributes(dummy_provider, DummyReceiver), "additional_argument")

    def test_copies_multiple_attributes(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class(
            common_attributes={"arbitrary_name1": ARBITRARY_VALUE, "arbitrary_name2": "arbitrary_string"}
        )

        assert copy_attributes(dummy_provider, DummyReceiver).arbitrary_name1 == dummy_provider.arbitrary_name1
        assert copy_attributes(dummy_provider, DummyReceiver).arbitrary_name2 == dummy_provider.arbitrary_name2

    def test_applies_custom_renaming(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class(
            only_provider={"provider_name": ARBITRARY_VALUE}, only_receiver={"receiver_name": "arbitrary_string"}
        )

        assert (
            copy_attributes(
                dummy_provider, DummyReceiver, conversions={"receiver_name": "provider_name"}
            ).receiver_name
            == dummy_provider.provider_name
        )

    def test_custom_renaming_fails_for_missing_receiving_counterpart(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class(
            only_provider={"provider_name": ARBITRARY_VALUE}
        )

        with pytest.raises(
            ValueError,
            match=f"Conversion failed. 'receiver_name' is not a member of the receiver <.*{CLASS_NAME}.*>. "
            "You gave {'receiver_name': 'provider_name'}.",
        ):
            copy_attributes(dummy_provider, DummyReceiver, conversions={"receiver_name": "provider_name"})

    def test_custom_renaming_fails_for_missing_provider_counterpart(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class(
            only_receiver={"receiver_name": ARBITRARY_VALUE}
        )

        with pytest.raises(
            ValueError,
            match=f"Conversion failed. 'provider_name' is not a member of the provider {dummy_provider}. "
            "You gave {'receiver_name': 'provider_name'}.",
        ):
            copy_attributes(dummy_provider, DummyReceiver, conversions={"receiver_name": "provider_name"})

    def test_custom_conversion_function(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class(
            common_attributes={"arbitrary_name": ARBITRARY_VALUE}
        )

        assert (
            copy_attributes(
                dummy_provider, DummyReceiver, conversions={"arbitrary_name": custom_conversion}
            ).arbitrary_name
            == custom_conversion(dummy_provider)
        )

    def test_removes_prefix(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class(
            only_provider={"prefixed_arbitrary_name": ARBITRARY_VALUE}, only_receiver={"arbitrary_name": None}
        )

        assert (
            copy_attributes(dummy_provider, DummyReceiver, remove_prefix="prefixed_").arbitrary_name
            == dummy_provider.prefixed_arbitrary_name
        )

    def test_ignore_attributes(self):
        dummy_provider, DummyReceiver = gen_provider_and_matching_receiver_class(
            common_attributes={"arbitrary_name": ARBITRARY_VALUE}
        )
        assert copy_attributes(dummy_provider, DummyReceiver, ignore=["arbitrary_name"]).arbitrary_name is None

    def test_copies_property_by_value(self):
        dummy_provider = ClassWithProperty()
        DummyReceiver = gen_class({"attribute": None})
        assert isinstance(
            getattr(copy_attributes(dummy_provider, DummyReceiver), "attribute"), type(dummy_provider._attribute)
        )


class TestConvertsTo(TestCase):
    def test_returns_same_class(self):
        DummyProvider, DummyReceiver = gen_two_classes()
        assert converts_to(DummyReceiver)(DummyProvider) is DummyProvider

    def test_adds_attribute_get_as_pypicongpu(self):
        DummyProvider, DummyReceiver = gen_two_classes()
        assert hasattr(converts_to(DummyReceiver)(DummyProvider), "get_as_pypicongpu")

    def test_fails_if_get_as_pypicongpu_exists(self):
        DummyProvider, DummyReceiver = gen_two_classes(only_receiver={"get_as_pypicongpu": ARBITRARY_VALUE})
        with pytest.raises(
            TypeError,
            match=f"Adding 'get_as_pypicongpu' failed because it already existed on receiving class .*{CLASS_NAME}.*.",
        ):
            converts_to(DummyReceiver)(DummyProvider)

    def _extract_relevant_members(self, lhs):
        return tuple(x for x in inspect.getmembers(lhs) if not x[0].startswith("__"))

    def assertInstancesEqual(self, lhs, rhs):
        assert type(lhs) == type(rhs)
        assert list(self._extract_relevant_members(lhs)) == list(self._extract_relevant_members(rhs))

    def test_get_as_pypicongpu_acts_like_copy_attributes(self):
        arbitrary_value2 = "arbitrary_string"
        arbitrary_value3 = 1234.5678
        arbitrary_value4 = True

        DummyProvider, DummyReceiver = gen_two_classes(
            common_attributes={"arbitrary_name": ARBITRARY_VALUE},
            only_provider={
                "prefixed_other_name": arbitrary_value4,
                "provider_name": arbitrary_value2,
                "_private_member": ARBITRARY_VALUE,
            },
            only_receiver={"receiver_name": arbitrary_value3, "other_name": None},
        )
        dummy_provider = converts_to(DummyReceiver, remove_prefix="prefixed_")(DummyProvider)()

        self.assertInstancesEqual(
            dummy_provider.get_as_pypicongpu(),
            copy_attributes(dummy_provider, DummyReceiver, remove_prefix="prefixed_"),
        )

    def test_handles_custom_conversions(self):
        arbitrary_value2 = "arbitrary_string"
        arbitrary_value3 = 1234.5678
        conversions = {"arbitrary_name": custom_conversion, "receiver_name": "provider_name"}

        DummyProvider, DummyReceiver = gen_two_classes(
            common_attributes={"arbitrary_name": ARBITRARY_VALUE},
            only_provider={"provider_name": arbitrary_value2, "_private_member": ARBITRARY_VALUE},
            only_receiver={"receiver_name": arbitrary_value3},
        )

        dummy_provider = converts_to(DummyReceiver, conversions=conversions)(DummyProvider)()
        self.assertInstancesEqual(
            dummy_provider.get_as_pypicongpu(), copy_attributes(dummy_provider, DummyReceiver, conversions=conversions)
        )

    def test_runs_a_preamble(self):
        preamble = MiniMock()
        DummyProvider, DummyReceiver = gen_two_classes()

        dummy_provider = converts_to(DummyReceiver, preamble=preamble)(DummyProvider)()
        assert preamble.counter == 0
        dummy_provider.get_as_pypicongpu()
        assert preamble.counter == 1

    def test_passes_through_arbitrary_arguments_to_conversions(self):
        conversions = {"arbitrary_name": lambda self, x: self.arbitrary_name - x}

        DummyProvider, DummyReceiver = gen_two_classes(
            common_attributes={"arbitrary_name": ARBITRARY_VALUE},
        )

        dummy_provider = converts_to(DummyReceiver, conversions=conversions)(DummyProvider)()
        assert (
            dummy_provider.get_as_pypicongpu(ARBITRARY_VALUE).arbitrary_name
            == conversions["arbitrary_name"](dummy_provider, ARBITRARY_VALUE)
        )

    def test_passes_through_arbitrary_arguments_to_preamble(self):
        preamble = MiniMock()
        DummyProvider, DummyReceiver = gen_two_classes()

        dummy_provider = converts_to(DummyReceiver, preamble=preamble)(DummyProvider)()
        assert preamble.args is None
        assert preamble.kwargs is None
        dummy_provider.get_as_pypicongpu(ARBITRARY_VALUE, arbitrary_kwarg=ARBITRARY_VALUE)
        assert preamble.args[1:] == (ARBITRARY_VALUE,)
        assert preamble.kwargs == {"arbitrary_kwarg": ARBITRARY_VALUE}

    def test_ignore_attributes(self):
        DummyProvider, DummyReceiver = gen_two_classes(common_attributes={"arbitrary_name": ARBITRARY_VALUE})
        dummy_provider = converts_to(DummyReceiver, ignore=["arbitrary_name"])(DummyProvider)()
        assert dummy_provider.get_as_pypicongpu().arbitrary_name is None

    def test_default_converts_to_uses_get_as_pypicongpu(self):
        DummyProvider, DummyReceiver = gen_two_classes(
            common_attributes={
                "arbitrary_name": gen_class({"get_as_pypicongpu": lambda self: ARBITRARY_VALUE}, use_values=True)()
            }
        )
        dummy_provider = default_converts_to(DummyReceiver)(DummyProvider)()
        assert (
            dummy_provider.get_as_pypicongpu().arbitrary_name == dummy_provider.arbitrary_name.get_as_pypicongpu()
        )
