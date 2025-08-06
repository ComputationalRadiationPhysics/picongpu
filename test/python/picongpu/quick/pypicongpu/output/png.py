"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output import Png, EMFieldScaleEnum, ColorScaleEnum
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
    def test_empty(self):
        """Empty or incomplete configurations are handled correctly."""
        png = Png()
        # Unset species
        with self.assertRaises(ValueError, match="species must be set"):
            png._get_serialized()

        # Set species but not period
        png.species = create_species()
        with self.assertRaises(ValueError, match="period must be set"):
            png._get_serialized()

        # Set species, period but not axis
        png.period = TimeStepSpec([slice(0, None, 100)])
        with self.assertRaises(ValueError, match="axis must be set"):
            png._get_serialized()

        # Set species, period, axis but not slicePoint
        png.axis = "xy"
        with self.assertRaises(ValueError, match="slicePoint must be set"):
            png._get_serialized()

        # Set species, period, axis, slicePoint but not folder
        png.slicePoint = 0.5
        with self.assertRaises(ValueError, match="folder must be set"):
            png._get_serialized()

        # Set species, period, axis, slicePoint, folder but not scale_image
        png.folder = "png_output"
        with self.assertRaises(ValueError, match="scale_image must be set"):
            png._get_serialized()

        # Set up to scale_to_cellsize
        png.scale_image = 1.0
        with self.assertRaises(ValueError, match="scale_to_cellsize must be set"):
            png._get_serialized()

        # Set up to white_box_per_GPU
        png.scale_to_cellsize = True
        with self.assertRaises(ValueError, match="white_box_per_GPU must be set"):
            png._get_serialized()

        # Set up to EM_FIELD_SCALE_CHANNEL1
        png.white_box_per_GPU = False
        with self.assertRaises(ValueError, match="EM_FIELD_SCALE_CHANNEL1 etc. must be set"):
            png._get_serialized()

        # Set up to preParticleDensCol (grouping channel parameters)
        png.EM_FIELD_SCALE_CHANNEL1 = EMFieldScaleEnum.AUTO
        png.EM_FIELD_SCALE_CHANNEL2 = EMFieldScaleEnum.PLASMA_WAVE
        png.EM_FIELD_SCALE_CHANNEL3 = EMFieldScaleEnum.CUSTOM
        with self.assertRaises(ValueError, match="preParticleDensCol and preChannel1Col etc. must be set"):
            png._get_serialized()

        # Set up to customNormalizationSI
        png.preParticleDensCol = ColorScaleEnum.RED
        png.preChannel1Col = ColorScaleEnum.GREEN
        png.preChannel2Col = ColorScaleEnum.BLUE
        png.preChannel3Col = ColorScaleEnum.GRAY
        with self.assertRaises(ValueError, match="customNormalizationSI must be set"):
            png._get_serialized()

        # Set up to preChannel1 (grouping opacity and channel parameters)
        png.customNormalizationSI = [1.0, 2.0, 3.0]
        png.preParticleDens_opacity = 0.8
        with self.assertRaises(ValueError, match="preChannel1_opacity and preChannel1 etc. must be set"):
            png._get_serialized()

        # Valid configuration
        png.preChannel1_opacity = 0.7
        png.preChannel2_opacity = 0.6
        png.preChannel3_opacity = 0.5
        png.preChannel1 = "field1"
        png.preChannel2 = "field2"
        png.preChannel3 = "field3"
        serialized = png._get_serialized()
        self.assertEqual(serialized["species"]["name"], "electron")
        self.assertEqual(serialized["period"]["specs"][0]["step"], 100)
        self.assertEqual(serialized["axis"], "xy")
        self.assertEqual(serialized["slicePoint"], 0.5)
        self.assertEqual(serialized["folder"], "png_output")
        self.assertEqual(serialized["scale_image"], 1.0)
        self.assertTrue(serialized["scale_to_cellsize"], "scale_to_cellsize should be True")
        self.assertFalse(serialized["white_box_per_GPU"], "white_box_per_GPU should be False")

    def test_types(self):
        """Type safety is ensured for all attributes."""
        png = Png()

        # Invalid species
        invalid_species = ["string", 1, 1.0, None, {}]
        for invalid in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                png.species = invalid

        # Invalid period
        invalid_periods = [1, "string", [], {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                png.period = invalid

        # Invalid axis
        invalid_axes = [1, 1.0, [], {}]
        for invalid in invalid_axes:
            with self.assertRaises(typeguard.TypeCheckError):
                png.axis = invalid

        # Invalid slicePoint
        invalid_slicePoints = ["string", [], {}, None]
        for invalid in invalid_slicePoints:
            with self.assertRaises(typeguard.TypeCheckError):
                png.slicePoint = invalid

        # Invalid folder
        invalid_folders = [1, 1.0, [], {}]
        for invalid in invalid_folders:
            with self.assertRaises(typeguard.TypeCheckError):
                png.folder = invalid

        # Invalid scale_image
        invalid_scale_images = ["string", [], {}, None]
        for invalid in invalid_scale_images:
            with self.assertRaises(typeguard.TypeCheckError):
                png.scale_image = invalid

        # Invalid scale_to_cellsize
        invalid_scale_to_cellsizes = ["string", 1.0, [], {}]
        for invalid in invalid_scale_to_cellsizes:
            with self.assertRaises(typeguard.TypeCheckError):
                png.scale_to_cellsize = invalid

        # Invalid white_box_per_GPU
        invalid_white_box_per_GPUs = ["string", 1.0, [], {}]
        for invalid in invalid_white_box_per_GPUs:
            with self.assertRaises(typeguard.TypeCheckError):
                png.white_box_per_GPU = invalid

        # Invalid EM_FIELD_SCALE_CHANNEL1/2/3
        invalid_em_field_scales = ["string", 1.0, [], {}, None]
        for invalid in invalid_em_field_scales:
            with self.assertRaises(ValueError):
                png.EM_FIELD_SCALE_CHANNEL1 = invalid
            with self.assertRaises(ValueError):
                png.EM_FIELD_SCALE_CHANNEL2 = invalid
            with self.assertRaises(ValueError):
                png.EM_FIELD_SCALE_CHANNEL3 = invalid

        # Invalid preParticleDensCol, preChannel1Col/2/3
        invalid_colors = [1, 1.0, [], {}, None]
        for invalid in invalid_colors:
            with self.assertRaises(ValueError):
                png.preParticleDensCol = invalid
            with self.assertRaises(ValueError):
                png.preChannel1Col = invalid
            with self.assertRaises(ValueError):
                png.preChannel2Col = invalid
            with self.assertRaises(ValueError):
                png.preChannel3Col = invalid

        # Invalid customNormalizationSI
        invalid_custom_normalizations = ["string", 1, 1.0, None]
        for invalid in invalid_custom_normalizations:
            with self.assertRaises(typeguard.TypeCheckError):
                png.customNormalizationSI = invalid

        # Invalid preParticleDens_opacity, preChannel1/2/3_opacity
        invalid_opacities = ["string", [], {}, None]
        for invalid in invalid_opacities:
            with self.assertRaises(typeguard.TypeCheckError):
                png.preParticleDens_opacity = invalid
            with self.assertRaises(typeguard.TypeCheckError):
                png.preChannel1_opacity = invalid
            with self.assertRaises(typeguard.TypeCheckError):
                png.preChannel2_opacity = invalid
            with self.assertRaises(typeguard.TypeCheckError):
                png.preChannel3_opacity = invalid

        # Invalid preChannel1/2/3
        invalid_channels = [1, 1.0, [], {}]
        for invalid in invalid_channels:
            with self.assertRaises(typeguard.TypeCheckError):
                png.preChannel1 = invalid
            with self.assertRaises(typeguard.TypeCheckError):
                png.preChannel2 = invalid
            with self.assertRaises(typeguard.TypeCheckError):
                png.preChannel3 = invalid

        # Valid configuration
        png.species = create_species()
        png.period = TimeStepSpec([slice(0, None, 100)])
        png.axis = "xy"
        png.slicePoint = 0.5
        png.folder = "png_output"
        png.scale_image = 1.0
        png.scale_to_cellsize = True
        png.white_box_per_GPU = False
        png.EM_FIELD_SCALE_CHANNEL1 = EMFieldScaleEnum.AUTO
        png.EM_FIELD_SCALE_CHANNEL2 = EMFieldScaleEnum.PLASMA_WAVE
        png.EM_FIELD_SCALE_CHANNEL3 = EMFieldScaleEnum.CUSTOM
        png.preParticleDensCol = ColorScaleEnum.RED
        png.preChannel1Col = ColorScaleEnum.GREEN
        png.preChannel2Col = ColorScaleEnum.BLUE
        png.preChannel3Col = ColorScaleEnum.GRAY
        png.customNormalizationSI = [1.0, 2.0, 3.0]
        png.preParticleDens_opacity = 0.8
        png.preChannel1_opacity = 0.7
        png.preChannel2_opacity = 0.6
        png.preChannel3_opacity = 0.5
        png.preChannel1 = "field1"
        png.preChannel2 = "field2"
        png.preChannel3 = "field3"
        png._get_serialized()  # Should succeed

    def test_rendering(self):
        """Serialized data is correctly formatted for template consumption."""
        png = Png()
        png.species = create_species()
        png.period = TimeStepSpec([slice(0, None, 100)])
        png.axis = "xy"
        png.slicePoint = 0.5
        png.folder = "png_output"
        png.scale_image = 1.0
        png.scale_to_cellsize = True
        png.white_box_per_GPU = False
        png.EM_FIELD_SCALE_CHANNEL1 = EMFieldScaleEnum.AUTO
        png.EM_FIELD_SCALE_CHANNEL2 = EMFieldScaleEnum.PLASMA_WAVE
        png.EM_FIELD_SCALE_CHANNEL3 = EMFieldScaleEnum.CUSTOM
        png.preParticleDensCol = ColorScaleEnum.RED
        png.preChannel1Col = ColorScaleEnum.GREEN
        png.preChannel2Col = ColorScaleEnum.BLUE
        png.preChannel3Col = ColorScaleEnum.GRAY
        png.customNormalizationSI = [1.0, 2.0, 3.0]
        png.preParticleDens_opacity = 0.8
        png.preChannel1_opacity = 0.7
        png.preChannel2_opacity = 0.6
        png.preChannel3_opacity = 0.5
        png.preChannel1 = "field1"
        png.preChannel2 = "field2"
        png.preChannel3 = "field3"

        context = png.get_rendering_context()
        self.assertTrue(context["typeID"]["png"], "typeID should be png")
        context = context["data"]
        self.assertEqual(context["species"]["name"], "electron")
        self.assertEqual(context["period"]["specs"][0]["step"], 100)
        self.assertEqual(context["axis"], "xy")
        self.assertEqual(context["slicePoint"], 0.5)
        self.assertEqual(context["folder"], "png_output")
        self.assertEqual(context["scale_image"], 1.0)
        self.assertTrue(context["scale_to_cellsize"], "scale_to_cellsize should be True")
        self.assertFalse(context["white_box_per_GPU"], "white_box_per_GPU should be False")
        self.assertEqual(context["EM_FIELD_SCALE_CHANNEL1"], -1, "EM_FIELD_SCALE_CHANNEL1 should be AUTO (-1)")
        self.assertEqual(context["EM_FIELD_SCALE_CHANNEL2"], 3, "EM_FIELD_SCALE_CHANNEL2 should be PLASMA_WAVE (3)")
        self.assertEqual(context["EM_FIELD_SCALE_CHANNEL3"], 6, "EM_FIELD_SCALE_CHANNEL3 should be CUSTOM (6)")
        self.assertEqual(context["preParticleDensCol"], "red")
        self.assertEqual(context["preChannel1Col"], "green")
        self.assertEqual(context["preChannel2Col"], "blue")
        self.assertEqual(context["preChannel3Col"], "gray")
        self.assertEqual(context["customNormalizationSI"], [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}])
        self.assertEqual(context["preParticleDens_opacity"], 0.8)
        self.assertEqual(context["preChannel1_opacity"], 0.7)
        self.assertEqual(context["preChannel2_opacity"], 0.6)
        self.assertEqual(context["preChannel3_opacity"], 0.5)
        self.assertEqual(context["preChannel1"], "field1")
        self.assertEqual(context["preChannel2"], "field2")
        self.assertEqual(context["preChannel3"], "field3")

        # Unset required attributes should fail
        png = Png()
        with self.assertRaises(ValueError, match="species must be set"):
            png.get_rendering_context()

    def test_validation(self):
        """Constraints on parameters are enforced."""
        png = Png()

        # Test unset parameters
        with self.assertRaises(ValueError, match="species must be set"):
            png.check()

        # Set all but customNormalizationSI
        png.species = create_species()
        png.period = TimeStepSpec([slice(0, None, 100)])
        png.axis = "xy"
        png.slicePoint = 0.5
        png.folder = "png_output"
        png.scale_image = 1.0
        png.scale_to_cellsize = True
        png.white_box_per_GPU = False
        png.EM_FIELD_SCALE_CHANNEL1 = EMFieldScaleEnum.AUTO
        png.EM_FIELD_SCALE_CHANNEL2 = EMFieldScaleEnum.PLASMA_WAVE
        png.EM_FIELD_SCALE_CHANNEL3 = EMFieldScaleEnum.CUSTOM
        png.preParticleDensCol = ColorScaleEnum.RED
        png.preChannel1Col = ColorScaleEnum.GREEN
        png.preChannel2Col = ColorScaleEnum.BLUE
        png.preChannel3Col = ColorScaleEnum.GRAY
        png.preParticleDens_opacity = 0.8
        png.preChannel1_opacity = 0.7
        png.preChannel2_opacity = 0.6
        png.preChannel3_opacity = 0.5
        png.preChannel1 = "field1"
        png.preChannel2 = "field2"
        png.preChannel3 = "field3"
        with self.assertRaises(ValueError, match="customNormalizationSI must be set"):
            png.check()

        # Set empty customNormalizationSI
        png.customNormalizationSI = []
        with self.assertRaises(ValueError, match="customNormalizationSI must not be empty"):
            png.check()

        # Valid configuration
        png.customNormalizationSI = [1.0, 2.0, 3.0]
        png.check()  # Should succeed
        serialized = png._get_serialized()
        self.assertEqual(serialized["species"]["name"], "electron")
        self.assertEqual(serialized["customNormalizationSI"], [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}])


if __name__ == "__main__":
    unittest.main()
