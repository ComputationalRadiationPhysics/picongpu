"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output.png import Png, EMFieldScaleEnum, ColorScaleEnum
from picongpu.pypicongpu.species import Species
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec
from picongpu.pypicongpu.species.attribute import Position, Momentum
import unittest
import typeguard


def create_species():
    """Helper function to create a valid Species object."""
    species = Species()
    species.name = "electron"
    species.attributes = [Position(), Momentum()]
    species.constants = []
    return species


class TestPng(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.species = create_species()
        self.period = TimeStepSpec([slice(0, None, 100)])
        self.valid_png = Png(
            species=self.species,
            period=self.period,
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            EM_FIELD_SCALE_CHANNEL1=EMFieldScaleEnum.AUTO,
            EM_FIELD_SCALE_CHANNEL2=EMFieldScaleEnum.PLASMA_WAVE,
            EM_FIELD_SCALE_CHANNEL3=EMFieldScaleEnum.CUSTOM,
            preParticleDensCol=ColorScaleEnum.RED,
            preChannel1Col=ColorScaleEnum.GREEN,
            preChannel2Col=ColorScaleEnum.BLUE,
            preChannel3Col=ColorScaleEnum.GRAY,
            customNormalizationSI=[1.0, 2.0, 3.0],
            preParticleDens_opacity=0.5,
            preChannel1_opacity=0.6,
            preChannel2_opacity=0.7,
            preChannel3_opacity=0.8,
            preChannel1="E_x",
            preChannel2="E_y",
            preChannel3="E_z",
        )

    def test_instantiation_and_types(self):
        """Test instantiation, type safety, and enum mapping."""
        # Valid configuration
        self.valid_png.check()
        serialized = self.valid_png._get_serialized()
        self.assertEqual(serialized["axis"], "xy")
        self.assertEqual(serialized["slicePoint"], 0.5)
        self.assertEqual(serialized["folder"], "output")

        # Type safety
        invalid_types = {
            "species": ["string", 1, 1.0, {}],
            "period": [13.2, [], "2", {}],
            "axis": [1, 1.0, {}, []],
            "slicePoint": ["string", {}, []],
            "folder": [1, 1.0, {}],
            "scale_image": ["string", {}, []],
            "scale_to_cellsize": ["string", 1.0, {}],
            "white_box_per_GPU": ["string", 1.0, {}],
            "EM_FIELD_SCALE_CHANNEL1": ["string", 1, 1.0, {}],
            "preParticleDensCol": ["invalid", 1, 1.0, {}],
            "customNormalizationSI": ["string", 1, 1.0, {}],
            "preParticleDens_opacity": ["string", {}, []],
            "preChannel1": [1, 1.0, {}, None],
        }
        for attr, invalid_values in invalid_types.items():
            for invalid in invalid_values:
                with self.subTest(attr=attr, value=invalid):
                    kwargs = {
                        "species": self.species,
                        "period": self.period,
                        "axis": "xy",
                        "slicePoint": 0.5,
                        "folder": "output",
                        "scale_image": 0.5,
                        "scale_to_cellsize": True,
                        "white_box_per_GPU": False,
                        "EM_FIELD_SCALE_CHANNEL1": EMFieldScaleEnum.AUTO,
                        "EM_FIELD_SCALE_CHANNEL2": EMFieldScaleEnum.PLASMA_WAVE,
                        "EM_FIELD_SCALE_CHANNEL3": EMFieldScaleEnum.CUSTOM,
                        "preParticleDensCol": ColorScaleEnum.RED,
                        "preChannel1Col": ColorScaleEnum.GREEN,
                        "preChannel2Col": ColorScaleEnum.BLUE,
                        "preChannel3Col": ColorScaleEnum.GRAY,
                        "customNormalizationSI": [1.0, 2.0, 3.0],
                        "preParticleDens_opacity": 0.5,
                        "preChannel1_opacity": 0.6,
                        "preChannel2_opacity": 0.7,
                        "preChannel3_opacity": 0.8,
                        "preChannel1": "E_x",
                        "preChannel2": "E_y",
                        "preChannel3": "E_z",
                    }
                    kwargs[attr] = invalid
                    with self.assertRaises((typeguard.TypeCheckError, ValueError)):
                        Png(**kwargs)

        # Enum string mapping
        png = Png(
            species=self.species,
            period=self.period,
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            EM_FIELD_SCALE_CHANNEL1=EMFieldScaleEnum(-1),  # Maps to AUTO
            EM_FIELD_SCALE_CHANNEL2=EMFieldScaleEnum.PLASMA_WAVE,
            EM_FIELD_SCALE_CHANNEL3=EMFieldScaleEnum.CUSTOM,
            preParticleDensCol=ColorScaleEnum("red"),  # Maps to RED
            preChannel1Col=ColorScaleEnum.GREEN,
            preChannel2Col=ColorScaleEnum.BLUE,
            preChannel3Col=ColorScaleEnum.GRAY,
            customNormalizationSI=[1.0, 2.0, 3.0],
            preParticleDens_opacity=0.5,
            preChannel1_opacity=0.6,
            preChannel2_opacity=0.7,
            preChannel3_opacity=0.8,
            preChannel1="E_x",
            preChannel2="E_y",
            preChannel3="E_z",
        )
        self.assertEqual(png.EM_FIELD_SCALE_CHANNEL1, EMFieldScaleEnum.AUTO)
        self.assertEqual(png.preParticleDensCol, ColorScaleEnum.RED)

    def test_validation_and_rendering(self):
        """Test validation constraints and serialization."""
        context = self.valid_png.get_rendering_context()
        self.assertTrue(context["typeID"]["png"])
        context = context["data"]
        self.assertEqual(context["period"]["specs"][0]["step"], 100)
        self.assertEqual(context["axis"], "xy")
        self.assertEqual(context["slicePoint"], 0.5)
        self.assertEqual(context["folder"], "output")
        self.assertEqual(context["scale_image"], 0.5)
        self.assertEqual(context["scale_to_cellsize"], True)
        self.assertEqual(context["white_box_per_GPU"], False)
        self.assertEqual(context["EM_FIELD_SCALE_CHANNEL1"], -1)
        self.assertEqual(context["preParticleDensCol"], "red")
        self.assertEqual(context["customNormalizationSI"], [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}])
        self.assertEqual(context["preChannel1"], "E_x")

        invalid_configs = [
            ({"axis": "xx"}, "axis must be 'xy', 'yx', 'xz', 'zx', 'yz', or 'zy'"),
            ({"slicePoint": 1.5}, "slicePoint must be in"),
            ({"scale_image": 0.0}, "scale_image must be positive"),
            (
                {"scale_image": 1.0, "scale_to_cellsize": True},
                "scale_image must not be 1.0 when scale_to_cellsize is True",
            ),
            ({"preParticleDens_opacity": 1.5}, "preParticleDens_opacity must be in"),
            ({"preChannel1": ""}, "preChannel1 must be a non-empty string"),
            ({"customNormalizationSI": [1.0, 2.0]}, "customNormalizationSI must contain exactly 3 floats"),
            ({"customNormalizationSI": [1.0, "2.0", 3.0]}, "customNormalizationSI values must be floats"),
        ]
        for invalid_config, error_msg in invalid_configs:
            with self.subTest(config=invalid_config):
                kwargs = {
                    "species": self.species,
                    "period": self.period,
                    "axis": "xy",
                    "slicePoint": 0.5,
                    "folder": "output",
                    "scale_image": 0.5,
                    "scale_to_cellsize": True,
                    "white_box_per_GPU": False,
                    "EM_FIELD_SCALE_CHANNEL1": EMFieldScaleEnum.AUTO,
                    "EM_FIELD_SCALE_CHANNEL2": EMFieldScaleEnum.PLASMA_WAVE,
                    "EM_FIELD_SCALE_CHANNEL3": EMFieldScaleEnum.CUSTOM,
                    "preParticleDensCol": ColorScaleEnum.RED,
                    "preChannel1Col": ColorScaleEnum.GREEN,
                    "preChannel2Col": ColorScaleEnum.BLUE,
                    "preChannel3Col": ColorScaleEnum.GRAY,
                    "customNormalizationSI": [1.0, 2.0, 3.0],
                    "preParticleDens_opacity": 0.5,
                    "preChannel1_opacity": 0.6,
                    "preChannel2_opacity": 0.7,
                    "preChannel3_opacity": 0.8,
                    "preChannel1": "E_x",
                    "preChannel2": "E_y",
                    "preChannel3": "E_z",
                }
                kwargs.update(invalid_config)
                png = Png(**kwargs)
                with self.assertRaisesRegex(ValueError, error_msg):
                    png.check()

        class TestPng(Png):
            def __init__(self):
                pass

        invalid_png = TestPng()
        with self.assertRaisesRegex(ValueError, "species must be set"):
            invalid_png.check()

        invalid_png = TestPng()
        invalid_png.species = self.species
        with self.assertRaisesRegex(ValueError, "period must be set"):
            invalid_png.check()

        class TestPngEMField(Png):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.__dict__["_EM_FIELD_SCALE_CHANNEL1"] = None

                def getter(_):
                    return None

                self.__class__.EM_FIELD_SCALE_CHANNEL1 = property(getter)

        png = TestPngEMField(
            **{
                "species": self.species,
                "period": self.period,
                "axis": "xy",
                "slicePoint": 0.5,
                "folder": "output",
                "scale_image": 0.5,
                "scale_to_cellsize": True,
                "white_box_per_GPU": False,
                "EM_FIELD_SCALE_CHANNEL1": EMFieldScaleEnum.AUTO,
                "EM_FIELD_SCALE_CHANNEL2": EMFieldScaleEnum.PLASMA_WAVE,
                "EM_FIELD_SCALE_CHANNEL3": EMFieldScaleEnum.CUSTOM,
                "preParticleDensCol": ColorScaleEnum.RED,
                "preChannel1Col": ColorScaleEnum.GREEN,
                "preChannel2Col": ColorScaleEnum.BLUE,
                "preChannel3Col": ColorScaleEnum.GRAY,
                "customNormalizationSI": [1.0, 2.0, 3.0],
                "preParticleDens_opacity": 0.5,
                "preChannel1_opacity": 0.6,
                "preChannel2_opacity": 0.7,
                "preChannel3_opacity": 0.8,
                "preChannel1": "E_x",
                "preChannel2": "E_y",
                "preChannel3": "E_z",
            }
        )
        with self.assertRaisesRegex(ValueError, "EM_FIELD_SCALE_CHANNEL1 must be in"):
            png.check()

        class TestPngColorScale(Png):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.__dict__["_preParticleDensCol"] = None

                def getter(_):
                    return None

                self.__class__.preParticleDensCol = property(getter)

        png = TestPngColorScale(
            **{
                "species": self.species,
                "period": self.period,
                "axis": "xy",
                "slicePoint": 0.5,
                "folder": "output",
                "scale_image": 0.5,
                "scale_to_cellsize": True,
                "white_box_per_GPU": False,
                "EM_FIELD_SCALE_CHANNEL1": EMFieldScaleEnum.AUTO,
                "EM_FIELD_SCALE_CHANNEL2": EMFieldScaleEnum.PLASMA_WAVE,
                "EM_FIELD_SCALE_CHANNEL3": EMFieldScaleEnum.CUSTOM,
                "preParticleDensCol": ColorScaleEnum.RED,
                "preChannel1Col": ColorScaleEnum.GREEN,
                "preChannel2Col": ColorScaleEnum.BLUE,
                "preChannel3Col": ColorScaleEnum.GRAY,
                "customNormalizationSI": [1.0, 2.0, 3.0],
                "preParticleDens_opacity": 0.5,
                "preChannel1_opacity": 0.6,
                "preChannel2_opacity": 0.7,
                "preChannel3_opacity": 0.8,
                "preChannel1": "E_x",
                "preChannel2": "E_y",
                "preChannel3": "E_z",
            }
        )
        with self.assertRaisesRegex(ValueError, "preParticleDensCol must be in"):
            png.check()

        png = Png(
            species=self.species,
            period=self.period,
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            EM_FIELD_SCALE_CHANNEL1=EMFieldScaleEnum.AUTO,
            EM_FIELD_SCALE_CHANNEL2=EMFieldScaleEnum.PLASMA_WAVE,
            EM_FIELD_SCALE_CHANNEL3=EMFieldScaleEnum.CUSTOM,
            preParticleDensCol=ColorScaleEnum.RED,
            preChannel1Col=ColorScaleEnum.GREEN,
            preChannel2Col=ColorScaleEnum.BLUE,
            preChannel3Col=ColorScaleEnum.GRAY,
            customNormalizationSI=[1.0, 2.0, 3.0],
            preParticleDens_opacity=0.5,
            preChannel1_opacity=0.6,
            preChannel2_opacity=0.7,
            preChannel3_opacity=0.8,
            preChannel1="field_E.x()",
            preChannel2="field_E.y() * field_E.y()",
            preChannel3="-1.0_X * field_B.z()",
        )
        png.check()


if __name__ == "__main__":
    unittest.main()
