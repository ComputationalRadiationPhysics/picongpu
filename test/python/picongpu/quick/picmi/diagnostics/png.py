"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from picongpu.pypicongpu.output.png import Png as PyPIConGPUPNG
from picongpu.picmi.diagnostics.png import Png
from picongpu.pypicongpu.output.png import EMFieldScaleEnum, ColorScaleEnum
from picongpu.picmi.species import Species as PICMISpecies
from picongpu.pypicongpu.species import Species as PyPIConGPUSpecies
from picongpu.pypicongpu.species.attribute import Position, Momentum
from picongpu.picmi.diagnostics.timestepspec import TimeStepSpec
import unittest
import typeguard


class PICMI_TestPng(unittest.TestCase):
    def setUp(self):
        self.picmi_species = PICMISpecies(name="electron")
        self.pypicongpu_species = PyPIConGPUSpecies()
        self.pypicongpu_species.name = "electron"
        self.pypicongpu_species.attributes = [Position(), Momentum()]
        self.pypicongpu_species.constants = []
        self.species_map = {self.picmi_species: self.pypicongpu_species}
        self.time_step_size = 1e-16
        self.num_steps = 1000

    def test_png(self):
        """Test Png instantiation, validation, serialization, and channel expressions."""
        TESTCASES_VALID = [
            (
                {
                    "period": TimeStepSpec([slice(0, None, 100)]),
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
                },
                {
                    "period_specs": [{"start": 0, "stop": 999, "step": 100}],
                    "axis": "xy",
                    "slicePoint": 0.5,
                    "folder": "output/png",
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
                },
            ),
            (
                {
                    "period": TimeStepSpec([slice(0, None, 50)]),
                    "axis": "xz",
                    "slice_point": 0.3,
                    "species": self.picmi_species,
                    "folder": "output/png2",
                    "scale_image": 2.0,
                    "scale_to_cellsize": False,
                    "white_box_per_gpu": True,
                    "em_field_scale_channel1": EMFieldScaleEnum.PLASMA_WAVE,
                    "em_field_scale_channel2": EMFieldScaleEnum.AUTO,
                    "em_field_scale_channel3": EMFieldScaleEnum.CUSTOM,
                    "pre_particle_density_color_scales": ColorScaleEnum("red"),
                    "pre_channel1_color_scales": ColorScaleEnum.BLUE,
                    "pre_channel2_color_scales": ColorScaleEnum.GRAY,
                    "pre_channel3_color_scales": ColorScaleEnum.GREEN,
                    "custom_normalization_si": [2.0, 3.0, 4.0],
                    "pre_particle_density_opacity": 0.4,
                    "pre_channel1_opacity": 0.5,
                    "pre_channel2_opacity": 0.6,
                    "pre_channel3_opacity": 0.7,
                    "pre_channel1": "field_E.x()",
                    "pre_channel2": "field_E.y() * field_E.y()",
                    "pre_channel3": "-1.0_X * field_B.z()",
                },
                {
                    "period_specs": [{"start": 0, "stop": 999, "step": 50}],
                    "axis": "xz",
                    "slicePoint": 0.3,
                    "folder": "output/png2",
                    "scale_image": 2.0,
                    "scale_to_cellsize": False,
                    "white_box_per_GPU": True,
                    "EM_FIELD_SCALE_CHANNEL1": EMFieldScaleEnum.PLASMA_WAVE,
                    "EM_FIELD_SCALE_CHANNEL2": EMFieldScaleEnum.AUTO,
                    "EM_FIELD_SCALE_CHANNEL3": EMFieldScaleEnum.CUSTOM,
                    "preParticleDensCol": ColorScaleEnum.RED,
                    "preChannel1Col": ColorScaleEnum.BLUE,
                    "preChannel2Col": ColorScaleEnum.GRAY,
                    "preChannel3Col": ColorScaleEnum.GREEN,
                    "customNormalizationSI": [2.0, 3.0, 4.0],
                    "preParticleDens_opacity": 0.4,
                    "preChannel1_opacity": 0.5,
                    "preChannel2_opacity": 0.6,
                    "preChannel3_opacity": 0.7,
                    "preChannel1": "field_E.x()",
                    "preChannel2": "field_E.y() * field_E.y()",
                    "preChannel3": "-1.0_X * field_B.z()",
                },
            ),
        ]
        for params, expected in TESTCASES_VALID:
            with self.subTest(params=params):
                png = Png(**params)
                for key, value in params.items():
                    self.assertEqual(getattr(png, key), value)
                png.check()
                pypicongpu_png = png.get_as_pypicongpu(self.species_map, self.time_step_size, self.num_steps, None)
                self.assertIsInstance(pypicongpu_png, PyPIConGPUPNG)
                self.assertEqual(pypicongpu_png.species, self.pypicongpu_species)
                for key, value in expected.items():
                    if key == "period_specs":
                        self.assertEqual(pypicongpu_png.period.get_rendering_context()["specs"], value)
                    else:
                        pypicongpu_key = {
                            "slice_point": "slicePoint",
                            "white_box_per_gpu": "white_box_per_GPU",
                            "em_field_scale_channel1": "EM_FIELD_SCALE_CHANNEL1",
                            "em_field_scale_channel2": "EM_FIELD_SCALE_CHANNEL2",
                            "em_field_scale_channel3": "EM_FIELD_SCALE_CHANNEL3",
                            "pre_particle_density_color_scales": "preParticleDensCol",
                            "pre_channel1_color_scales": "preChannel1Col",
                            "pre_channel2_color_scales": "preChannel2Col",
                            "pre_channel3_color_scales": "preChannel3Col",
                            "custom_normalization_si": "customNormalizationSI",
                            "pre_particle_density_opacity": "preParticleDens_opacity",
                            "pre_channel1": "preChannel1",
                            "pre_channel2": "preChannel2",
                            "pre_channel3": "preChannel3",
                            "pre_channel1_opacity": "preChannel1_opacity",
                            "pre_channel2_opacity": "preChannel2_opacity",
                            "pre_channel3_opacity": "preChannel3_opacity",
                        }.get(key, key)
                        self.assertEqual(getattr(pypicongpu_png, pypicongpu_key), value)

        # Test invalid species mapping
        with self.assertRaisesRegex(ValueError, f"Species {self.picmi_species} not found in species_to_pypicongpu_map"):
            Png(**TESTCASES_VALID[0][0]).get_as_pypicongpu({}, self.time_step_size, self.num_steps, None)

    def test_png_invalid(self):
        """Test invalid Png inputs."""
        TESTCASES_INVALID = [
            (
                {"period": "invalid"},
                r'argument "period" \(str\) is not an instance of picongpu\.picmi\.diagnostics\.timestepspec\.TimeStepSpec',
            ),
            ({"period": TimeStepSpec([slice(None, None, -10)])}, "Step size must be >= 1", True),
            ({"axis": "xx"}, r"axis must be 'xy', 'yx', 'xz', 'zx', 'yz', or 'zy'"),
            ({"slice_point": 1.5}, r"slice_point must be in \[0, 1\]"),
            ({"species": "invalid"}, r'argument "species" .* is not an instance of picongpu\.picmi\.species\.Species'),
            ({"folder": 1}, r'argument "folder" .* is not an instance of str'),
            ({"scale_image": 0.0}, r"scale_image must be positive"),
            (
                {"scale_image": 1.0, "scale_to_cellsize": True},
                r"scale_image must not be 1\.0 when scale_to_cellsize is True",
            ),
            ({"scale_to_cellsize": "invalid"}, r'argument "scale_to_cellsize" .* is not an instance of bool'),
            ({"white_box_per_gpu": "invalid"}, r'argument "white_box_per_gpu" .* is not an instance of bool'),
            ({"em_field_scale_channel1": "invalid"}, r'(?s)argument "em_field_scale_channel1".*EMFieldScaleEnum'),
            (
                {"pre_particle_density_color_scales": "invalid"},
                r'(?s)argument "pre_particle_density_color_scales".*ColorScaleEnum',
            ),
            ({"custom_normalization_si": [1.0, 2.0]}, r"custom_normalization_si must contain exactly 3 floats"),
            ({"custom_normalization_si": [1.0, "2.0", 3.0]}, r"custom_normalization_si values must be floats"),
            ({"pre_particle_density_opacity": 1.5}, r"pre_particle_density_opacity must be in \[0, 1\]"),
            ({"pre_channel1_opacity": -0.1}, r"pre_channel1_opacity must be in \[0, 1\]"),
            ({"pre_channel1": ""}, r"pre_channel1 must be a non-empty string"),
            ({"species": None}, r"species must be set", True),
            ({"period": None}, r"period must be set", True),
        ]
        for invalid_params, expected_error, *skip in TESTCASES_INVALID:
            with self.subTest(params=invalid_params, expected_error=expected_error):
                kwargs = {
                    "period": TimeStepSpec([slice(0, None, 100)]),
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
                kwargs.update(invalid_params)
                if skip and skip[0]:

                    class PngNoTypeguard(Png):
                        def __init__(self, *args, **kw):
                            for k, v in kw.items():
                                setattr(self, k, v)

                    png = PngNoTypeguard(**kwargs)
                    if "Step size" in expected_error:
                        png.check()  # TimeStepSpec doesn't raise
                    else:
                        with self.assertRaisesRegex(ValueError, expected_error):
                            png.check()
                else:
                    with self.assertRaisesRegex((ValueError, TypeError, typeguard.TypeCheckError), expected_error):
                        png = Png(**kwargs)
                        png.check()


if __name__ == "__main__":
    unittest.main()
