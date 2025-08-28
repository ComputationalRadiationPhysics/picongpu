"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.picmi.diagnostics.openpmd_sources.source_base import SourceBase
import unittest
import typeguard
import typing


@typeguard.typechecked
class MockSource(SourceBase):
    """Mock implementation of SourceBase for testing."""

    def __init__(self, filter_value: typing.Optional[str] = "all"):
        self._filter = filter_value

    @property
    def filter(self) -> typing.Optional[str]:
        return self._filter

    def check(self) -> None:
        valid_filters = ["all", "custom_filter"]
        if self._filter is not None and not isinstance(self._filter, str):
            raise ValueError(f"Filter must be a string or None, got {type(self._filter)}")
        if self._filter is not None and self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")

    def get_as_pypicongpu(self) -> typing.Any:
        return {"mock_source": {"filter": self._filter}}


class PICMI_TestSourceBase(unittest.TestCase):
    def test_source_base_abstract(self):
        """Test that SourceBase cannot be instantiated directly."""
        with self.assertRaises(TypeError, msg="Can't instantiate abstract class SourceBase"):
            SourceBase()

    def test_mock_source_instantiation(self):
        """Test MockSource instantiation and filter property."""
        # Valid cases
        source = MockSource(filter_value="all")
        self.assertEqual(source.filter, "all")
        source.check()  # Should not raise

        source = MockSource(filter_value=None)
        self.assertIsNone(source.filter)
        source.check()  # Should not raise

        source = MockSource(filter_value="custom_filter")
        self.assertEqual(source.filter, "custom_filter")
        source.check()  # Should not raise

        # Invalid filter value
        with self.assertRaisesRegex(ValueError, r"Filter must be one of \['all', 'custom_filter'\], got invalid"):
            source = MockSource(filter_value="invalid")
            source.check()

    def test_mock_source_get_as_pypicongpu(self):
        """Test MockSource get_as_pypicongpu method."""
        source = MockSource(filter_value="all")
        pypicongpu_source = source.get_as_pypicongpu()
        self.assertEqual(pypicongpu_source, {"mock_source": {"filter": "all"}})

        source = MockSource(filter_value=None)
        pypicongpu_source = source.get_as_pypicongpu()
        self.assertEqual(pypicongpu_source, {"mock_source": {"filter": None}})

    def test_typeguard_enforcement(self):
        """Test that typeguard enforces filter type."""
        with self.assertRaisesRegex(
            typeguard.TypeCheckError, r"argument \"filter_value\" \(int\) did not match any element in the union"
        ):
            MockSource(filter_value=123)  # Should raise before check()


if __name__ == "__main__":
    unittest.main()
