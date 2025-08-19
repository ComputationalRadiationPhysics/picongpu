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
        """Set up a valid Png configuration for tests."""
        self.valid_png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

    def test_empty(self):
        """Invalid configurations are handled correctly."""
        # Invalid axis
        with self.assertRaisesRegex(ValueError, "axis must be 'xy', 'xz', or 'yz'"):
            png = Png(
                species=create_species(),
                period=TimeStepSpec([slice(0, None, 100)]),
                axis="xx",
                slicePoint=0.5,
                folder="output",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_GPU=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
            png._get_serialized()

        # Invalid slicePoint
        with self.assertRaisesRegex(ValueError, "slicePoint must be in"):
            png = Png(
                species=create_species(),
                period=TimeStepSpec([slice(0, None, 100)]),
                axis="xy",
                slicePoint=1.5,
                folder="output",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_GPU=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
            png._get_serialized()

        # Invalid scale_image
        with self.assertRaisesRegex(ValueError, "scale_image must be positive"):
            png = Png(
                species=create_species(),
                period=TimeStepSpec([slice(0, None, 100)]),
                axis="xy",
                slicePoint=0.5,
                folder="output",
                scale_image=0.0,
                scale_to_cellsize=True,
                white_box_per_GPU=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
            png._get_serialized()

        # Invalid scale_image with scale_to_cellsize
        with self.assertRaisesRegex(ValueError, "scale_image must not be 1.0 when scale_to_cellsize is True"):
            png = Png(
                species=create_species(),
                period=TimeStepSpec([slice(0, None, 100)]),
                axis="xy",
                slicePoint=0.5,
                folder="output",
                scale_image=1.0,
                scale_to_cellsize=True,
                white_box_per_GPU=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
            png._get_serialized()

        # Valid configuration with empty folder
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=0.5,
            folder="",
            scale_image=1.0,
            scale_to_cellsize=False,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
        serialized = png._get_serialized()
        self.assertEqual(serialized["axis"], "xy")
        self.assertEqual(serialized["slicePoint"], 0.5)
        self.assertEqual(serialized["folder"], "")

    def test_types(self):
        """Type safety is ensured for all attributes."""
        # Invalid species
        invalid_species = ["string", 1, 1.0, {}]
        for invalid in invalid_species:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=invalid,
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

        # Invalid period
        invalid_periods = [13.2, [], "2", {}]
        for invalid in invalid_periods:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=create_species(),
                    period=invalid,
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

        # Invalid axis (non-string types)
        invalid_axes_non_string = [1, 1.0, {}, []]
        for invalid in invalid_axes_non_string:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis=invalid,
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

        # Invalid axis (invalid strings)
        invalid_axes_strings = ["x", "xyz", "xx"]
        for invalid in invalid_axes_strings:
            with self.assertRaisesRegex(ValueError, "axis must be 'xy', 'xz', or 'yz'"):
                png = Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis=invalid,
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
                png.check()

        # Invalid slicePoint
        invalid_slicePoints = ["string", {}, []]
        for invalid in invalid_slicePoints:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=invalid,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

        # Invalid folder
        invalid_folders = [1, 1.0, {}]
        for invalid in invalid_folders:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder=invalid,
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

        # Invalid scale_image
        invalid_scale_images = ["string", {}, []]
        for invalid in invalid_scale_images:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=invalid,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

        # Invalid scale_to_cellsize
        invalid_scale_to_cellsizes = ["string", 1.0, {}]
        for invalid in invalid_scale_to_cellsizes:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=invalid,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

        # Invalid white_box_per_GPU
        invalid_white_boxes = ["string", 1.0, {}]
        for invalid in invalid_white_boxes:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=invalid,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

        # Invalid EM_FIELD_SCALE_CHANNEL1
        invalid_scales = ["string", 1, 1.0, {}]
        for invalid in invalid_scales:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=invalid,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

        # Invalid preParticleDensCol
        invalid_colors = ["invalid", 1, 1.0, {}]
        for invalid in invalid_colors:
            with self.assertRaises((typeguard.TypeCheckError, ValueError)):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                    preParticleDensCol=invalid,
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

        # Invalid customNormalizationSI
        invalid_normalizations = ["string", 1, 1.0, {}]
        for invalid in invalid_normalizations:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                    preParticleDensCol=ColorScaleEnum.RED,
                    preChannel1Col=ColorScaleEnum.GREEN,
                    preChannel2Col=ColorScaleEnum.BLUE,
                    preChannel3Col=ColorScaleEnum.GRAY,
                    customNormalizationSI=invalid,
                    preParticleDens_opacity=0.5,
                    preChannel1_opacity=0.6,
                    preChannel2_opacity=0.7,
                    preChannel3_opacity=0.8,
                    preChannel1="E_x",
                    preChannel2="E_y",
                    preChannel3="E_z",
                )

        # Invalid preParticleDens_opacity
        invalid_opacities = ["string", {}, []]
        for invalid in invalid_opacities:
            with self.assertRaises(typeguard.TypeCheckError):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                    preParticleDensCol=ColorScaleEnum.RED,
                    preChannel1Col=ColorScaleEnum.GREEN,
                    preChannel2Col=ColorScaleEnum.BLUE,
                    preChannel3Col=ColorScaleEnum.GRAY,
                    customNormalizationSI=[1.0, 2.0, 3.0],
                    preParticleDens_opacity=invalid,
                    preChannel1_opacity=0.6,
                    preChannel2_opacity=0.7,
                    preChannel3_opacity=0.8,
                    preChannel1="E_x",
                    preChannel2="E_y",
                    preChannel3="E_z",
                )

        # Test invalid preChannel inputs
        invalid_non_strings = [1, 1.0, {}, None]
        invalid_empty_string = [""]

        # Test non-string inputs (caught by typeguard)
        for invalid in invalid_non_strings:
            with self.assertRaisesRegex(typeguard.TypeCheckError, "is not an instance of str"):
                Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                    preParticleDensCol=ColorScaleEnum.RED,
                    preChannel1Col=ColorScaleEnum.GREEN,
                    preChannel2Col=ColorScaleEnum.BLUE,
                    preChannel3Col=ColorScaleEnum.GRAY,
                    customNormalizationSI=[1.0, 2.0, 3.0],
                    preParticleDens_opacity=0.5,
                    preChannel1_opacity=0.6,
                    preChannel2_opacity=0.7,
                    preChannel3_opacity=0.8,
                    preChannel1=invalid,
                    preChannel2="field_E.y()",
                    preChannel3="-1.0_X * field_E.y()",
                )

        # Test empty string input (caught by custom validation)
        for invalid in invalid_empty_string:
            with self.assertRaisesRegex(ValueError, "preChannel1 must be a non-empty string"):
                png = Png(
                    species=create_species(),
                    period=TimeStepSpec([slice(0, None, 100)]),
                    axis="xy",
                    slicePoint=0.5,
                    folder="output",
                    scale_image=0.5,
                    scale_to_cellsize=True,
                    white_box_per_GPU=False,
                    em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                    em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                    em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                    preParticleDensCol=ColorScaleEnum.RED,
                    preChannel1Col=ColorScaleEnum.GREEN,
                    preChannel2Col=ColorScaleEnum.BLUE,
                    preChannel3Col=ColorScaleEnum.GRAY,
                    customNormalizationSI=[1.0, 2.0, 3.0],
                    preParticleDens_opacity=0.5,
                    preChannel1_opacity=0.6,
                    preChannel2_opacity=0.7,
                    preChannel3_opacity=0.8,
                    preChannel1=invalid,
                    preChannel2="field_E.y()",
                    preChannel3="-1.0_X * field_E.y()",
                )
                png.check()

        # Test EMFieldScaleEnum string mapping
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum(-1),  # Maps to AUTO
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
        self.assertEqual(png.EM_FIELD_SCALE_CHANNEL1, EMFieldScaleEnum.AUTO)

        # Test ColorScaleEnum string mapping
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
        self.assertEqual(png.preParticleDensCol, ColorScaleEnum.RED)

        # Valid configuration
        self.valid_png._get_serialized()  # Should succeed

    def test_rendering(self):
        """Serialized data is correctly formatted for template consumption."""
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

    def test_validation(self):
        """Constraints on parameters are enforced."""
        # Invalid axis
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xx",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
        with self.assertRaisesRegex(ValueError, "axis must be 'xy', 'xz', or 'yz'"):
            png.check()

        # Invalid slicePoint
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=1.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
        with self.assertRaisesRegex(ValueError, "slicePoint must be in"):
            png.check()

        # Invalid scale_image
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.0,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
        with self.assertRaisesRegex(ValueError, "scale_image must be positive"):
            png.check()

        # Invalid scale_image with scale_to_cellsize
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=1.0,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
        with self.assertRaisesRegex(ValueError, "scale_image must not be 1.0 when scale_to_cellsize is True"):
            png.check()

        # Invalid preParticleDens_opacity
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
            preParticleDensCol=ColorScaleEnum.RED,
            preChannel1Col=ColorScaleEnum.GREEN,
            preChannel2Col=ColorScaleEnum.BLUE,
            preChannel3Col=ColorScaleEnum.GRAY,
            customNormalizationSI=[1.0, 2.0, 3.0],
            preParticleDens_opacity=1.5,
            preChannel1_opacity=0.6,
            preChannel2_opacity=0.7,
            preChannel3_opacity=0.8,
            preChannel1="E_x",
            preChannel2="E_y",
            preChannel3="E_z",
        )
        with self.assertRaisesRegex(ValueError, "preParticleDens_opacity must be in"):
            png.check()

        # Invalid preChannel1 (empty string)
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
            preParticleDensCol=ColorScaleEnum.RED,
            preChannel1Col=ColorScaleEnum.GREEN,
            preChannel2Col=ColorScaleEnum.BLUE,
            preChannel3Col=ColorScaleEnum.GRAY,
            customNormalizationSI=[1.0, 2.0, 3.0],
            preParticleDens_opacity=0.5,
            preChannel1_opacity=0.6,
            preChannel2_opacity=0.7,
            preChannel3_opacity=0.8,
            preChannel1="",  # Invalid: empty string
            preChannel2="field_E.y()",
            preChannel3="-1.0_X * field_E.y()",
        )
        with self.assertRaisesRegex(ValueError, "preChannel1 must be a non-empty string"):
            png.check()

        # Invalid EMFieldScaleEnum
        with self.assertRaises(ValueError):
            Png(
                species=create_species(),
                period=TimeStepSpec([slice(0, None, 100)]),
                axis="xy",
                slicePoint=0.5,
                folder="output",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_GPU=False,
                em_field_scale_channel1=EMFieldScaleEnum(999),  # Invalid value
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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

        # Invalid ColorScaleEnum
        with self.assertRaises(ValueError):
            Png(
                species=create_species(),
                period=TimeStepSpec([slice(0, None, 100)]),
                axis="xy",
                slicePoint=0.5,
                folder="output",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_GPU=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                preParticleDensCol=ColorScaleEnum("invalid"),  # Invalid value
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

        # Valid configuration
        self.valid_png.check()  # Should succeed
        serialized = self.valid_png._get_serialized()
        self.assertEqual(serialized["axis"], "xy")
        self.assertEqual(serialized["slicePoint"], 0.5)
        self.assertEqual(serialized["folder"], "output")

    def test_channels(self):
        """Validate preChannel* field components."""
        # Invalid preChannel1 (empty string)
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
            preParticleDensCol=ColorScaleEnum.RED,
            preChannel1Col=ColorScaleEnum.GREEN,
            preChannel2Col=ColorScaleEnum.BLUE,
            preChannel3Col=ColorScaleEnum.GRAY,
            customNormalizationSI=[1.0, 2.0, 3.0],
            preParticleDens_opacity=0.5,
            preChannel1_opacity=0.6,
            preChannel2_opacity=0.7,
            preChannel3_opacity=0.8,
            preChannel1="",
            preChannel2="E_y",
            preChannel3="E_z",
        )
        with self.assertRaisesRegex(ValueError, "preChannel1 must be a non-empty string"):
            png.check()

        # Valid channels
        png = Png(
            species=create_species(),
            period=TimeStepSpec([slice(0, None, 100)]),
            axis="xy",
            slicePoint=0.5,
            folder="output",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_GPU=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
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
        png.check()  # Should succeed


if __name__ == "__main__":
    unittest.main()
