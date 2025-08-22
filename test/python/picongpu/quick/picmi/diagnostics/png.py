"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output.png import Png as PyPIConGPUPNG
from picongpu.pypicongpu.output.png import EMFieldScaleEnum, ColorScaleEnum
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
from picongpu.pypicongpu.species.attribute import Position, Momentum
from picongpu.pypicongpu.output.timestepspec import TimeStepSpec as PyPIConGPUTimeStepSpec
from picongpu.picmi.diagnostics.png import Png
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
import unittest
import typeguard


class PICMI_TestPng(unittest.TestCase):
    def setUp(self):
        """Set up common test fixtures."""
        self.picmi_species = PICMISpecies(name="electron")
        self.pypicongpu_species = PyPIConGPUSpecies()
        self.pypicongpu_species.name = "electron"
        self.pypicongpu_species.attributes = [Position(), Momentum()]
        self.pypicongpu_species.constants = []
        self.species_map = {self.picmi_species: self.pypicongpu_species}
        self.period = TimeStepSpec([slice(0, None, 100)])
        self.time_step_size = 1e-16  # Example time step in seconds
        self.num_steps = 1000  # Example number of steps
        self.simulation_box = None  # Not used, included for signature
        self.valid_png = Png(
            period=self.period,
            axis="xy",
            slice_point=0.5,
            species=self.picmi_species,
            folder="output/png",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_gpu=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
            pre_particle_density_color_scales=ColorScaleEnum.RED,
            pre_channel1_color_scales=ColorScaleEnum.GREEN,
            pre_channel2_color_scales=ColorScaleEnum.BLUE,
            pre_channel3_color_scales=ColorScaleEnum.GRAY,
            custom_normalization_si=[1.0, 2.0, 3.0],
            pre_particle_density_opacity=0.5,
            pre_channel1_opacity=0.6,
            pre_channel2_opacity=0.7,
            pre_channel3_opacity=0.8,
            pre_channel1="E_x",
            pre_channel2="E_y",
            pre_channel3="E_z",
        )

    def test_instantiation(self):
        """Test valid instantiation of Png class."""
        png = self.valid_png
        self.assertEqual(png.axis, "xy")
        self.assertEqual(png.slice_point, 0.5)
        self.assertEqual(png.species, self.picmi_species)
        self.assertEqual(png.folder, "output/png")
        self.assertEqual(png.scale_image, 0.5)
        self.assertTrue(png.scale_to_cellsize)
        self.assertFalse(png.white_box_per_gpu)
        self.assertEqual(png.em_field_scale_channel1, EMFieldScaleEnum.AUTO)
        self.assertEqual(png.em_field_scale_channel2, EMFieldScaleEnum.PLASMA_WAVE)
        self.assertEqual(png.em_field_scale_channel3, EMFieldScaleEnum.CUSTOM)
        self.assertEqual(png.pre_particle_density_color_scales, ColorScaleEnum.RED)
        self.assertEqual(png.pre_channel1_color_scales, ColorScaleEnum.GREEN)
        self.assertEqual(png.pre_channel2_color_scales, ColorScaleEnum.BLUE)
        self.assertEqual(png.pre_channel3_color_scales, ColorScaleEnum.GRAY)
        self.assertEqual(png.custom_normalization_si, [1.0, 2.0, 3.0])
        self.assertEqual(png.pre_particle_density_opacity, 0.5)
        self.assertEqual(png.pre_channel1_opacity, 0.6)
        self.assertEqual(png.pre_channel2_opacity, 0.7)
        self.assertEqual(png.pre_channel3_opacity, 0.8)
        self.assertEqual(png.pre_channel1, "E_x")
        self.assertEqual(png.pre_channel2, "E_y")
        self.assertEqual(png.pre_channel3, "E_z")
        png.check()  # Should not raise

    def test_type_safety(self):
        """Test type safety for all attributes."""
        invalid_types = {
            "period": [13.2, [], "2", {}],
            "axis": [1, 1.0, {}, []],
            "slice_point": ["string", {}, []],
            "species": ["string", 1, 1.0, {}],
            "folder": [1, 1.0, {}],
            "scale_image": ["string", {}, []],
            "scale_to_cellsize": ["string", 1.0, {}],
            "white_box_per_gpu": ["string", 1.0, {}],
            "em_field_scale_channel1": ["string", 1, 1.0, {}],
            "em_field_scale_channel2": ["string", 1, 1.0, {}],
            "em_field_scale_channel3": ["string", 1, 1.0, {}],
            "pre_particle_density_color_scales": ["invalid", 1, 1.0, {}],
            "pre_channel1_color_scales": ["invalid", 1, 1.0, {}],
            "pre_channel2_color_scales": ["invalid", 1, 1.0, {}],
            "pre_channel3_color_scales": ["invalid", 1, 1.0, {}],
            "custom_normalization_si": ["string", 1, 1.0, {}],
            "pre_particle_density_opacity": ["string", {}, []],
            "pre_channel1_opacity": ["string", {}, []],
            "pre_channel2_opacity": ["string", {}, []],
            "pre_channel3_opacity": ["string", {}, []],
            "pre_channel1": [1, 1.0, {}, None],
            "pre_channel2": [1, 1.0, {}, None],
            "pre_channel3": [1, 1.0, {}, None],
        }

        for attr, invalid_values in invalid_types.items():
            for invalid in invalid_values:
                kwargs = {
                    "period": self.period,
                    "axis": "xy",
                    "slice_point": 0.5,
                    "species": self.picmi_species,
                    "folder": "output/png",
                    "scale_image": 0.5,
                    "scale_to_cellsize": True,
                    "white_box_per_gpu": False,
                    "em_field_scale_channel1": EMFieldScaleEnum.AUTO,
                    "em_field_scale_channel2": EMFieldScaleEnum.PLASMA_WAVE,
                    "em_field_scale_channel3": EMFieldScaleEnum.CUSTOM,
                    "pre_particle_density_color_scales": ColorScaleEnum.RED,
                    "pre_channel1_color_scales": ColorScaleEnum.GREEN,
                    "pre_channel2_color_scales": ColorScaleEnum.BLUE,
                    "pre_channel3_color_scales": ColorScaleEnum.GRAY,
                    "custom_normalization_si": [1.0, 2.0, 3.0],
                    "pre_particle_density_opacity": 0.5,
                    "pre_channel1_opacity": 0.6,
                    "pre_channel2_opacity": 0.7,
                    "pre_channel3_opacity": 0.8,
                    "pre_channel1": "E_x",
                    "pre_channel2": "E_y",
                    "pre_channel3": "E_z",
                }
                kwargs[attr] = invalid
                with self.assertRaises(
                    typeguard.TypeCheckError, msg=f"Type check failed for {attr} with value {invalid}"
                ):
                    Png(**kwargs)

    def test_validation(self):
        """Test validation of constraints in check()."""
        # Invalid axis
        with self.assertRaisesRegex(ValueError, "axis must be 'xy', 'xz', or 'yz'"):
            png = Png(
                period=self.period,
                axis="xx",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Invalid slice_point
        with self.assertRaisesRegex(ValueError, "slice_point must be in"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=1.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Invalid scale_image
        with self.assertRaisesRegex(ValueError, "scale_image must be positive"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.0,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Invalid scale_image with scale_to_cellsize
        with self.assertRaisesRegex(ValueError, "scale_image must not be 1.0 when scale_to_cellsize is True"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=1.0,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Invalid custom_normalization_si length
        with self.assertRaisesRegex(ValueError, "custom_normalization_si must contain exactly 3 floats"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Invalid custom_normalization_si values
        with self.assertRaisesRegex(ValueError, "custom_normalization_si values must be floats"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, "2.0", 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Invalid pre_particle_density_opacity
        with self.assertRaisesRegex(ValueError, "pre_particle_density_opacity must be in"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=1.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Invalid pre_channel1_opacity
        with self.assertRaisesRegex(ValueError, "pre_channel1_opacity must be in"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=-0.1,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Invalid pre_channel1 (empty string)
        with self.assertRaisesRegex(ValueError, "pre_channel1 must be a non-empty string"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Invalid em_field_scale_channel1
        with self.assertRaisesRegex(ValueError, "em_field_scale_channel1 must be in"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=None,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Invalid pre_particle_density_color_scales
        with self.assertRaisesRegex(ValueError, "pre_particle_density_color_scales must be in"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=None,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Unset species
        class PngNoTypeguard(Png):
            def __init__(self, *args, **kwargs):
                # Directly set attributes to bypass typeguard
                self.period = kwargs.get("period")
                self.axis = kwargs.get("axis")
                self.slice_point = kwargs.get("slice_point")
                self.species = kwargs.get("species")
                self.folder = kwargs.get("folder")
                self.scale_image = kwargs.get("scale_image")
                self.scale_to_cellsize = kwargs.get("scale_to_cellsize")
                self.white_box_per_gpu = kwargs.get("white_box_per_gpu")
                self.em_field_scale_channel1 = kwargs.get("em_field_scale_channel1")
                self.em_field_scale_channel2 = kwargs.get("em_field_scale_channel2")
                self.em_field_scale_channel3 = kwargs.get("em_field_scale_channel3")
                self.pre_particle_density_color_scales = kwargs.get("pre_particle_density_color_scales")
                self.pre_channel1_color_scales = kwargs.get("pre_channel1_color_scales")
                self.pre_channel2_color_scales = kwargs.get("pre_channel2_color_scales")
                self.pre_channel3_color_scales = kwargs.get("pre_channel3_color_scales")
                self.custom_normalization_si = kwargs.get("custom_normalization_si")
                self.pre_particle_density_opacity = kwargs.get("pre_particle_density_opacity")
                self.pre_channel1_opacity = kwargs.get("pre_channel1_opacity")
                self.pre_channel2_opacity = kwargs.get("pre_channel2_opacity")
                self.pre_channel3_opacity = kwargs.get("pre_channel3_opacity")
                self.pre_channel1 = kwargs.get("pre_channel1")
                self.pre_channel2 = kwargs.get("pre_channel2")
                self.pre_channel3 = kwargs.get("pre_channel3")

        with self.assertRaisesRegex(ValueError, "species must be set"):
            png = PngNoTypeguard(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=None,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Unset period
        class PngNoTypeguard(Png):
            def __init__(self, *args, **kwargs):
                # Directly set attributes to bypass typeguard
                self.period = kwargs.get("period")
                self.axis = kwargs.get("axis")
                self.slice_point = kwargs.get("slice_point")
                self.species = kwargs.get("species")
                self.folder = kwargs.get("folder")
                self.scale_image = kwargs.get("scale_image")
                self.scale_to_cellsize = kwargs.get("scale_to_cellsize")
                self.white_box_per_gpu = kwargs.get("white_box_per_gpu")
                self.em_field_scale_channel1 = kwargs.get("em_field_scale_channel1")
                self.em_field_scale_channel2 = kwargs.get("em_field_scale_channel2")
                self.em_field_scale_channel3 = kwargs.get("em_field_scale_channel3")
                self.pre_particle_density_color_scales = kwargs.get("pre_particle_density_color_scales")
                self.pre_channel1_color_scales = kwargs.get("pre_channel1_color_scales")
                self.pre_channel2_color_scales = kwargs.get("pre_channel2_color_scales")
                self.pre_channel3_color_scales = kwargs.get("pre_channel3_color_scales")
                self.custom_normalization_si = kwargs.get("custom_normalization_si")
                self.pre_particle_density_opacity = kwargs.get("pre_particle_density_opacity")
                self.pre_channel1_opacity = kwargs.get("pre_channel1_opacity")
                self.pre_channel2_opacity = kwargs.get("pre_channel2_opacity")
                self.pre_channel3_opacity = kwargs.get("pre_channel3_opacity")
                self.pre_channel1 = kwargs.get("pre_channel1")
                self.pre_channel2 = kwargs.get("pre_channel2")
                self.pre_channel3 = kwargs.get("pre_channel3")

        with self.assertRaisesRegex(ValueError, "period must be set"):
            png = PngNoTypeguard(
                period=None,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="E_x",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

        # Valid configuration
        self.valid_png.check()

    def test_get_as_pypicongpu(self):
        """Test conversion to PyPIConGPU format."""
        pypicongpu_png = self.valid_png.get_as_pypicongpu(
            self.species_map, self.time_step_size, self.num_steps, self.simulation_box
        )
        self.assertIsInstance(pypicongpu_png, PyPIConGPUPNG)
        self.assertEqual(pypicongpu_png.species, self.pypicongpu_species)
        self.assertEqual(pypicongpu_png.axis, "xy")
        self.assertEqual(pypicongpu_png.slicePoint, 0.5)
        self.assertEqual(pypicongpu_png.folder, "output/png")
        self.assertEqual(pypicongpu_png.scale_image, 0.5)
        self.assertEqual(pypicongpu_png.scale_to_cellsize, True)
        self.assertEqual(pypicongpu_png.white_box_per_GPU, False)
        self.assertEqual(pypicongpu_png.EM_FIELD_SCALE_CHANNEL1, EMFieldScaleEnum.AUTO)
        self.assertEqual(pypicongpu_png.EM_FIELD_SCALE_CHANNEL2, EMFieldScaleEnum.PLASMA_WAVE)
        self.assertEqual(pypicongpu_png.EM_FIELD_SCALE_CHANNEL3, EMFieldScaleEnum.CUSTOM)
        self.assertEqual(pypicongpu_png.preParticleDensCol, ColorScaleEnum.RED)
        self.assertEqual(pypicongpu_png.preChannel1Col, ColorScaleEnum.GREEN)
        self.assertEqual(pypicongpu_png.preChannel2Col, ColorScaleEnum.BLUE)
        self.assertEqual(pypicongpu_png.preChannel3Col, ColorScaleEnum.GRAY)
        self.assertEqual(pypicongpu_png.customNormalizationSI, [1.0, 2.0, 3.0])
        self.assertEqual(pypicongpu_png.preParticleDens_opacity, 0.5)
        self.assertEqual(pypicongpu_png.preChannel1_opacity, 0.6)
        self.assertEqual(pypicongpu_png.preChannel2_opacity, 0.7)
        self.assertEqual(pypicongpu_png.preChannel3_opacity, 0.8)
        self.assertEqual(pypicongpu_png.preChannel1, "E_x")
        self.assertEqual(pypicongpu_png.preChannel2, "E_y")
        self.assertEqual(pypicongpu_png.preChannel3, "E_z")

        # Invalid species mapping
        invalid_map = {}
        with self.assertRaisesRegex(ValueError, "Species .* not found in species_to_pypicongpu_map"):
            self.valid_png.get_as_pypicongpu(invalid_map, self.time_step_size, self.num_steps, self.simulation_box)

        # Valid period conversion
        pypicongpu_period = self.period.get_as_pypicongpu(self.time_step_size, self.num_steps)
        self.assertIsInstance(pypicongpu_period, PyPIConGPUTimeStepSpec)
        self.assertEqual(pypicongpu_period.get_rendering_context()["specs"][0]["step"], 100)

    def test_channels(self):
        """Test validation of pre_channel* field components."""
        png = Png(
            period=self.period,
            axis="xy",
            slice_point=0.5,
            species=self.picmi_species,
            folder="output/png",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_gpu=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
            pre_particle_density_color_scales=ColorScaleEnum.RED,
            pre_channel1_color_scales=ColorScaleEnum.GREEN,
            pre_channel2_color_scales=ColorScaleEnum.BLUE,
            pre_channel3_color_scales=ColorScaleEnum.GRAY,
            custom_normalization_si=[1.0, 2.0, 3.0],
            pre_particle_density_opacity=0.5,
            pre_channel1_opacity=0.6,
            pre_channel2_opacity=0.7,
            pre_channel3_opacity=0.8,
            pre_channel1="field_E.x()",
            pre_channel2="field_E.y() * field_E.y()",
            pre_channel3="-1.0_X * field_B.z()",
        )
        png.check()
        pypicongpu_png = png.get_as_pypicongpu(
            self.species_map, self.time_step_size, self.num_steps, self.simulation_box
        )
        self.assertEqual(pypicongpu_png.preChannel1, "field_E.x()")
        self.assertEqual(pypicongpu_png.preChannel2, "field_E.y() * field_E.y()")
        self.assertEqual(pypicongpu_png.preChannel3, "-1.0_X * field_B.z()")

        # Invalid empty channel
        with self.assertRaisesRegex(ValueError, "pre_channel1 must be a non-empty string"):
            png = Png(
                period=self.period,
                axis="xy",
                slice_point=0.5,
                species=self.picmi_species,
                folder="output/png",
                scale_image=0.5,
                scale_to_cellsize=True,
                white_box_per_gpu=False,
                em_field_scale_channel1=EMFieldScaleEnum.AUTO,
                em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
                em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
                pre_particle_density_color_scales=ColorScaleEnum.RED,
                pre_channel1_color_scales=ColorScaleEnum.GREEN,
                pre_channel2_color_scales=ColorScaleEnum.BLUE,
                pre_channel3_color_scales=ColorScaleEnum.GRAY,
                custom_normalization_si=[1.0, 2.0, 3.0],
                pre_particle_density_opacity=0.5,
                pre_channel1_opacity=0.6,
                pre_channel2_opacity=0.7,
                pre_channel3_opacity=0.8,
                pre_channel1="",
                pre_channel2="E_y",
                pre_channel3="E_z",
            )
            png.check()

    def test_enum_string_mapping(self):
        """Test string mapping for Enum types."""
        png = Png(
            period=self.period,
            axis="xy",
            slice_point=0.5,
            species=self.picmi_species,
            folder="output/png",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_gpu=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
            pre_particle_density_color_scales=ColorScaleEnum.RED,
            pre_channel1_color_scales=ColorScaleEnum.GREEN,
            pre_channel2_color_scales=ColorScaleEnum.BLUE,
            pre_channel3_color_scales=ColorScaleEnum.GRAY,
            custom_normalization_si=[1.0, 2.0, 3.0],
            pre_particle_density_opacity=0.5,
            pre_channel1_opacity=0.6,
            pre_channel2_opacity=0.7,
            pre_channel3_opacity=0.8,
            pre_channel1="E_x",
            pre_channel2="E_y",
            pre_channel3="E_z",
        )
        self.assertEqual(png.em_field_scale_channel1, EMFieldScaleEnum.AUTO)
        pypicongpu_png = png.get_as_pypicongpu(
            self.species_map, self.time_step_size, self.num_steps, self.simulation_box
        )
        self.assertEqual(pypicongpu_png.EM_FIELD_SCALE_CHANNEL1, EMFieldScaleEnum.AUTO)

        # ColorScaleEnum string mapping
        png = Png(
            period=self.period,
            axis="xy",
            slice_point=0.5,
            species=self.picmi_species,
            folder="output/png",
            scale_image=0.5,
            scale_to_cellsize=True,
            white_box_per_gpu=False,
            em_field_scale_channel1=EMFieldScaleEnum.AUTO,
            em_field_scale_channel2=EMFieldScaleEnum.PLASMA_WAVE,
            em_field_scale_channel3=EMFieldScaleEnum.CUSTOM,
            pre_particle_density_color_scales=ColorScaleEnum("red"),  # Maps to RED
            pre_channel1_color_scales=ColorScaleEnum.GREEN,
            pre_channel2_color_scales=ColorScaleEnum.BLUE,
            pre_channel3_color_scales=ColorScaleEnum.GRAY,
            custom_normalization_si=[1.0, 2.0, 3.0],
            pre_particle_density_opacity=0.5,
            pre_channel1_opacity=0.6,
            pre_channel2_opacity=0.7,
            pre_channel3_opacity=0.8,
            pre_channel1="E_x",
            pre_channel2="E_y",
            pre_channel3="E_z",
        )
        self.assertEqual(png.pre_particle_density_color_scales, ColorScaleEnum.RED)
        pypicongpu_png = png.get_as_pypicongpu(
            self.species_map, self.time_step_size, self.num_steps, self.simulation_box
        )
        self.assertEqual(pypicongpu_png.preParticleDensCol, ColorScaleEnum.RED)


if __name__ == "__main__":
    unittest.main()
