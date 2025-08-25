import unittest
from picongpu.pypicongpu.output import TimeStepSpec


class TestTimeStepSpec(unittest.TestCase):
    def test_init_with_valid_slices(self):
        specs = [slice(0, 10, 1), slice(10, 20, 2)]
        time_step_spec = TimeStepSpec(specs)
        self.assertEqual(time_step_spec.specs, specs)

    def test_init_with_invalid_slices(self):
        time_step_spec = TimeStepSpec([slice(0, 10, 1), "invalid"])
        with self.assertRaises(ValueError):
            time_step_spec._get_serialized()  # must call the serialization which triggers error

    def test_serialize_with_valid_slices(self):
        specs = [slice(0, 10, 1), slice(10, 20, 2)]
        time_step_spec = TimeStepSpec(specs)
        serialized = time_step_spec._get_serialized()
        expected = {
            "specs": [
                {"start": 0, "stop": 10, "step": 1},
                {"start": 10, "stop": 20, "step": 2},
            ]
        }
        self.assertEqual(serialized, expected)

    def test_serialize_with_none_values(self):
        specs = [slice(None, 10, 1), slice(10, None, 2), slice(10, 20, None)]
        time_step_spec = TimeStepSpec(specs)
        serialized = time_step_spec._get_serialized()
        expected = {
            "specs": [
                {"start": 0, "stop": 10, "step": 1},
                {"start": 10, "stop": -1, "step": 2},
                {"start": 10, "stop": 20, "step": 1},
            ]
        }
        self.assertEqual(serialized, expected)

    def get_rendering_context(self):
        return self._get_serialized()
