"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.png import Png as PyPIConGPUPNG
from ...pypicongpu.species.species import Species as PyPIConGPUSpecies
from ...pypicongpu.output.png import EMFieldScaleEnum, ColorScaleEnum
from ..species import Species as PICMISpecies
from .timestepspec import TimeStepSpec

import typeguard
from typing import List, Dict, Optional


@typeguard.typechecked
class Png:
    """
    Specifies the parameters for PNG output in PIConGPU via PICMI interface.

    This plugin generates 2D PNG images of field and particle data.

    Parameters
    ----------
    period: TimeStepSpec
        Specify on which time steps the plugin should run.
        Unit: steps (simulation time steps).

    axis: string
        Axis combination for the 2D slice (e.g., "xy", "xz", "yz").

    slice_point: float
        Ratio for the slice position in the dimension not used in axis (0.0 to 1.0).
        Unit: dimensionless.

    species: PICMISpecies
        Particle species to include in the PNG output (e.g., electron, proton).

    folder: string
        Folder where the PNGs will be stored.

    scale_image: float
        Scaling factor applied to the image before writing to file.
        Unit: dimensionless.

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

    custom_normalization_si: List[float]
        Custom normalization factors for B (T), E (V/m), and current (A) when using EMFieldScaleEnum.CUSTOM.
        Unit: T, V/m, A.

    pre_particle_density_opacity: float
        Opacity of the particle density overlay (0.0 to 1.0).
        Unit: dimensionless.

    pre_channel1_opacity: float
        Opacity for channel 1 data (0.0 to 1.0).
        Unit: dimensionless.

    pre_channel2_opacity: float
        Opacity for channel 2 data (0.0 to 1.0).
        Unit: dimensionless.

    pre_channel3_opacity: float
        Opacity for channel 3 data (0.0 to 1.0).
        Unit: dimensionless.

    pre_channel1: string
        Field component for channel 1 (e.g., "E_x").

    pre_channel2: string
        Field component for channel 2 (e.g., "E_y").

    pre_channel3: string
        Field component for channel 3 (e.g., "E_z").
    """

    def __init__(
        self,
        period: TimeStepSpec,
        axis: str,
        slice_point: float,
        species: PICMISpecies,
        folder: str,
        scale_image: float,
        scale_to_cellsize: bool,
        white_box_per_gpu: bool,
        em_field_scale_channel1: Optional[EMFieldScaleEnum],
        em_field_scale_channel2: Optional[EMFieldScaleEnum],
        em_field_scale_channel3: Optional[EMFieldScaleEnum],
        pre_particle_density_color_scales: Optional[ColorScaleEnum],
        pre_channel1_color_scales: Optional[ColorScaleEnum],
        pre_channel2_color_scales: Optional[ColorScaleEnum],
        pre_channel3_color_scales: Optional[ColorScaleEnum],
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
        self.folder = folder
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

    def check(self):
        """
        Check if the parameters are valid.

        Raises
        ------
        ValueError
            If any parameter is invalid.
        """
        if self.species is None:
            raise ValueError("species must be set")
        if self.period is None:
            raise ValueError("period must be set")
        if self.axis not in ["xy", "yx", "xz", "zx", "yz", "zy"]:
            raise ValueError(f"axis must be 'xy', 'yx', 'xz', 'zx', 'yz', or 'zy', got {self.axis}")
        if self.slice_point < 0.0 or self.slice_point > 1.0:
            raise ValueError(f"slice_point must be in [0, 1], got {self.slice_point}")
        if self.scale_image <= 0:
            raise ValueError(f"scale_image must be positive, got {self.scale_image}")
        if self.scale_to_cellsize and self.scale_image == 1.0:
            raise ValueError(f"scale_image must not be 1.0 when scale_to_cellsize is True, got {self.scale_image}")
        if self.pre_particle_density_opacity < 0 or self.pre_particle_density_opacity > 1:
            raise ValueError(f"pre_particle_density_opacity must be in [0, 1], got {self.pre_particle_density_opacity}")
        if self.pre_channel1_opacity < 0 or self.pre_channel1_opacity > 1:
            raise ValueError(f"pre_channel1_opacity must be in [0, 1], got {self.pre_channel1_opacity}")
        if self.pre_channel2_opacity < 0 or self.pre_channel2_opacity > 1:
            raise ValueError(f"pre_channel2_opacity must be in [0, 1], got {self.pre_channel2_opacity}")
        if self.pre_channel3_opacity < 0 or self.pre_channel3_opacity > 1:
            raise ValueError(f"pre_channel3_opacity must be in [0, 1], got {self.pre_channel3_opacity}")
        for channel, name in [
            (self.pre_channel1, "pre_channel1"),
            (self.pre_channel2, "pre_channel2"),
            (self.pre_channel3, "pre_channel3"),
        ]:
            if not isinstance(channel, str) or not channel.strip():
                raise ValueError(f"{name} must be a non-empty string, got {channel}")
        if len(self.custom_normalization_si) != 3:
            raise ValueError(
                f"custom_normalization_si must contain exactly 3 floats, got {len(self.custom_normalization_si)}"
            )
        for val in self.custom_normalization_si:
            if not isinstance(val, float):
                raise ValueError(f"custom_normalization_si values must be floats, got {val}")
        if not isinstance(self.em_field_scale_channel1, EMFieldScaleEnum):
            raise ValueError(
                f"em_field_scale_channel1 must be in {list(EMFieldScaleEnum)}, got {self.em_field_scale_channel1}"
            )
        if not isinstance(self.em_field_scale_channel2, EMFieldScaleEnum):
            raise ValueError(
                f"em_field_scale_channel2 must be in {list(EMFieldScaleEnum)}, got {self.em_field_scale_channel2}"
            )
        if not isinstance(self.em_field_scale_channel3, EMFieldScaleEnum):
            raise ValueError(
                f"em_field_scale_channel3 must be in {list(EMFieldScaleEnum)}, got {self.em_field_scale_channel3}"
            )
        if not isinstance(self.pre_particle_density_color_scales, ColorScaleEnum):
            raise ValueError(
                f"pre_particle_density_color_scales must be in {list(ColorScaleEnum)}, got {self.pre_particle_density_color_scales}"
            )
        if not isinstance(self.pre_channel1_color_scales, ColorScaleEnum):
            raise ValueError(
                f"pre_channel1_color_scales must be in {list(ColorScaleEnum)}, got {self.pre_channel1_color_scales}"
            )
        if not isinstance(self.pre_channel2_color_scales, ColorScaleEnum):
            raise ValueError(
                f"pre_channel2_color_scales must be in {list(ColorScaleEnum)}, got {self.pre_channel2_color_scales}"
            )
        if not isinstance(self.pre_channel3_color_scales, ColorScaleEnum):
            raise ValueError(
                f"pre_channel3_color_scales must be in {list(ColorScaleEnum)}, got {self.pre_channel3_color_scales}"
            )

    def get_as_pypicongpu(
        self,
        species_to_pypicongpu_map: Dict[PICMISpecies, PyPIConGPUSpecies],
        time_step_size: float,
        num_steps: int,
        simulation_box=None,
    ) -> PyPIConGPUPNG:
        self.check()
        if self.species not in species_to_pypicongpu_map:
            raise ValueError(f"Species {self.species} not found in species_to_pypicongpu_map")
        pypicongpu_species = species_to_pypicongpu_map[self.species]
        pypicongpu_period = self.period.get_as_pypicongpu(time_step_size, num_steps)

        pypicongpu_png = PyPIConGPUPNG(
            species=pypicongpu_species,
            period=pypicongpu_period,
            axis=self.axis,
            slicePoint=self.slice_point,
            folder=self.folder,
            scale_image=self.scale_image,
            scale_to_cellsize=self.scale_to_cellsize,
            white_box_per_GPU=self.white_box_per_gpu,
            EM_FIELD_SCALE_CHANNEL1=self.em_field_scale_channel1,
            EM_FIELD_SCALE_CHANNEL2=self.em_field_scale_channel2,
            EM_FIELD_SCALE_CHANNEL3=self.em_field_scale_channel3,
            preParticleDensCol=self.pre_particle_density_color_scales,
            preChannel1Col=self.pre_channel1_color_scales,
            preChannel2Col=self.pre_channel2_color_scales,
            preChannel3Col=self.pre_channel3_color_scales,
            customNormalizationSI=self.custom_normalization_si,
            preParticleDens_opacity=self.pre_particle_density_opacity,
            preChannel1_opacity=self.pre_channel1_opacity,
            preChannel2_opacity=self.pre_channel2_opacity,
            preChannel3_opacity=self.pre_channel3_opacity,
            preChannel1=self.pre_channel1,
            preChannel2=self.pre_channel2,
            preChannel3=self.pre_channel3,
        )
        return pypicongpu_png
