"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.png import Png as PyPIConGPUPNG
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies
from ...pypicongpu.output.png import EMFieldScaleEnum, ColorScaleEnum

from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec

import typeguard
from typing import List


@typeguard.typechecked
class Png:
    """
    Specifies the parameters for PNG output in PIConGPU.

    This plugin generates 2D PNG images of field and particle data.

    Parameters
    ----------
    period: TimeStepSpec
        Specify on which time steps the plugin should run.
        Unit: steps (simulation time steps).

    axis: string
        Axis combination for the 2D slice (e.g., "yx").

    slice_point: float
        Ratio for the slice position in the dimension not used in axis (e.g., "z") (0.0 to 1.0).
        [unit: dimensionless]

    species: string
        Name of the particle species to count (e.g., "electron", "proton").

    folder_name: string
        Folder name where the PNGs will be stored.

    scale_image: float
        Scaling factor applied to the image before writing to file.
        [unit: dimensionless]

    scale_to_cellsize: bool
        Whether to scale the image to account for non-quadratic cell sizes.

    white_box_per_gpu: bool
        If true, draws white lines indicating GPU boundaries.

    em_field_scale_channel1: EMFieldScaleEnum
        Scaling mode for EM fields in channel 1.

    em_field_scale_channel2: EMFieldScaleEnum
        Scaling mode for EM fields in channel 2.

    em_field_scale_channel3: EMFieldScaleEnum
        Scaling mode for EM fields in channel 3.

    pre_particle_density_color_scales: ColorScaleEnum
        Color scale for particle density.

    pre_channel1_color_scales: ColorScaleEnum
        Color scale for channel 1.

    pre_channel2_color_scales: ColorScaleEnum
        Color scale for channel 2.

    pre_channel3_color_scales: ColorScaleEnum
        Color scale for channel 3.

    custom_normalization_si: list of 3 floats
        Custom normalization factors for B (T), E (V/m), and current (A) (when using scale mode 6).
        [unit: T, V/m, A]

    pre_particle_density_opacity: float
        Opacity of the particle density overlay (0.0 to 1.0).
        [unit: dimensionless]

    pre_channel1_opacity: float
        Opacity for channel 1 data (0.0 to 1.0).
        [unit: dimensionless]

    pre_channel2_opacity: float
        Opacity for channel 2 data (0.0 to 1.0).
        [unit: dimensionless]

    pre_channel3_opacity: float
        Opacity for channel 3 data (0.0 to 1.0).
        [unit: dimensionless]

    pre_channel1: string
        Custom expression for channel 1.

    pre_channel2: string
        Custom expression for channel 2.

    pre_channel3: string
        Custom expression for channel 3.
    """

    def check(self):
        if not (0.0 <= self.slice_point <= 1.0):
            raise ValueError("Slice point must be between 0.0 and 1.0")

        if not (0.0 <= self.pre_particle_density_opacity <= 1.0):
            raise ValueError("pre particle density opacity must be between 0.0 and 1.0")
        if not (0.0 <= self.pre_channel1_opacity <= 1.0):
            raise ValueError("Pre channel 1 opacity must be between 0.0 and 1.0")
        if not (0.0 <= self.pre_channel2_opacity <= 1.0):
            raise ValueError("Pre channel 2 opacity must be between 0.0 and 1.0")
        if not (0.0 <= self.pre_channel3_opacity <= 1.0):
            raise ValueError("Pre channel 3 opacity must be between 0.0 and 1.0")

        # Validate EM field scaling for channels
        if self.em_field_scale_channel1 not in EMFieldScaleEnum:
            raise ValueError(f"Invalid EM field scale for channel 1. Valid options are {list(EMFieldScaleEnum)}.")
        if self.em_field_scale_channel2 not in EMFieldScaleEnum:
            raise ValueError(f"Invalid EM field scale for channel 2. Valid options are {list(EMFieldScaleEnum)}.")
        if self.em_field_scale_channel3 not in EMFieldScaleEnum:
            raise ValueError(f"Invalid EM field scale for channel 3. Valid options are {list(EMFieldScaleEnum)}.")

        # Validate color scales for particle density and channels
        if self.pre_particle_density_color_scales not in ColorScaleEnum:
            raise ValueError(f"Invalid color scale for particle density. Valid options are {list(ColorScaleEnum)}.")
        if self.pre_channel1_color_scales not in ColorScaleEnum:
            raise ValueError(f"Invalid color scale for channel 1. Valid options are {list(ColorScaleEnum)}.")
        if self.pre_channel2_color_scales not in ColorScaleEnum:
            raise ValueError(f"Invalid color scale for channel 2. Valid options are {list(ColorScaleEnum)}.")
        if self.pre_channel3_color_scales not in ColorScaleEnum:
            raise ValueError(f"Invalid color scale for channel 3. Valid options are {list(ColorScaleEnum)}.")

    def __init__(
        self,
        species: PICMISpecies,
        period: TimeStepSpec,
        axis: str,
        slice_point: float,
        folder_name: str,
        scale_image: float,
        scale_to_cellsize: bool,
        white_box_per_gpu: bool,
        em_field_scale_channel1: EMFieldScaleEnum,
        em_field_scale_channel2: EMFieldScaleEnum,
        em_field_scale_channel3: EMFieldScaleEnum,
        pre_particle_density_color_scales: ColorScaleEnum,
        pre_channel1_color_scales: ColorScaleEnum,
        pre_channel2_color_scales: ColorScaleEnum,
        pre_channel3_color_scales: ColorScaleEnum,
        custom_normalization_si: List[float],
        pre_particle_density_opacity: float,
        pre_channel1_opacity: float,
        pre_channel2_opacity: float,
        pre_channel3_opacity: float,
        pre_channel1: str,
        pre_channel2: str,
        pre_channel3: str,
    ):
        self.period = period
        self.axis = axis
        self.slice_point = slice_point
        self.species = species
        self.folder_name = folder_name
        self.scale_image = scale_image
        self.scale_to_cellsize = scale_to_cellsize
        self.white_box_per_gpu = white_box_per_gpu
        self.em_field_scale_channel1 = em_field_scale_channel1
        self.em_field_scale_channel2 = em_field_scale_channel2
        self.em_field_scale_channel3 = em_field_scale_channel3
        self.pre_particle_density_color_scales = pre_particle_density_color_scales
        self.pre_channel1_color_scales = pre_channel1_color_scales
        self.pre_channel2_color_scales = pre_channel2_color_scales
        self.pre_channel3_color_scales = pre_channel3_color_scales
        self.custom_normalization_si = custom_normalization_si
        self.pre_particle_density_opacity = pre_particle_density_opacity
        self.pre_channel1_opacity = pre_channel1_opacity
        self.pre_channel2_opacity = pre_channel2_opacity
        self.pre_channel3_opacity = pre_channel3_opacity
        self.pre_channel1 = pre_channel1
        self.pre_channel2 = pre_channel2
        self.pre_channel3 = pre_channel3

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size,
        num_steps,
        simulation_box=None,  # Added to match OpenPMD signature, not used
    ) -> PyPIConGPUPNG:
        self.check()

        if self.species not in dict_species_picmi_to_pypicongpu.keys():
            raise ValueError(f"Species {self.species} is not known to Simulation")

        pypicongpu_species = dict_species_picmi_to_pypicongpu.get(self.species)

        if pypicongpu_species is None:
            raise ValueError(f"Species {self.species} is not mapped to a PyPIConGPUSpecies.")

        pypicongpu_png = PyPIConGPUPNG()
        pypicongpu_png.period = self.period.get_as_pypicongpu(time_step_size, num_steps)
        pypicongpu_png.axis = self.axis
        pypicongpu_png.slicePoint = self.slice_point
        pypicongpu_png.species = pypicongpu_species
        pypicongpu_png.folder = self.folder_name
        pypicongpu_png.scale_image = self.scale_image
        pypicongpu_png.scale_to_cellsize = self.scale_to_cellsize
        pypicongpu_png.white_box_per_GPU = self.white_box_per_gpu
        pypicongpu_png.EM_FIELD_SCALE_CHANNEL1 = self.em_field_scale_channel1
        pypicongpu_png.EM_FIELD_SCALE_CHANNEL2 = self.em_field_scale_channel2
        pypicongpu_png.EM_FIELD_SCALE_CHANNEL3 = self.em_field_scale_channel3
        pypicongpu_png.preParticleDensCol = self.pre_particle_density_color_scales
        pypicongpu_png.preChannel1Col = self.pre_channel1_color_scales
        pypicongpu_png.preChannel2Col = self.pre_channel2_color_scales
        pypicongpu_png.preChannel3Col = self.pre_channel3_color_scales
        pypicongpu_png.customNormalizationSI = self.custom_normalization_si
        pypicongpu_png.preParticleDens_opacity = self.pre_particle_density_opacity
        pypicongpu_png.preChannel1_opacity = self.pre_channel1_opacity
        pypicongpu_png.preChannel2_opacity = self.pre_channel2_opacity
        pypicongpu_png.preChannel3_opacity = self.pre_channel3_opacity
        pypicongpu_png.preChannel1 = self.pre_channel1
        pypicongpu_png.preChannel2 = self.pre_channel2
        pypicongpu_png.preChannel3 = self.pre_channel3

        return pypicongpu_png
