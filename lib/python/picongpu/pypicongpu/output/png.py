"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .. import util
from ..species import Species
from .plugin import Plugin
from .timestepspec import TimeStepSpec

import typeguard
import typing
from enum import Enum


class EMFieldScaleEnum(Enum):
    AUTO = -1
    PLASMA_WAVE = 3
    CUSTOM = 6
    INCIDENT = 7

    @classmethod
    def _missing_(cls, value):
        """Ensure strings map correctly to Enum values."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


class ColorScaleEnum(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    GRAY = "gray"
    GRAY_INV = "grayInv"
    NONE = "none"

    @classmethod
    def _missing_(cls, value):
        """Ensure strings map correctly to Enum values."""
        for member in cls:
            if member.value == value:
                return member
        raise ValueError(f"{value} is not a valid {cls.__name__}")


@typeguard.typechecked
class Png(Plugin):
    species = util.build_typesafe_property(Species)
    period = util.build_typesafe_property(TimeStepSpec)
    axis = util.build_typesafe_property(str)
    slicePoint = util.build_typesafe_property(float)
    folder = util.build_typesafe_property(str)
    scale_image = util.build_typesafe_property(float)
    scale_to_cellsize = util.build_typesafe_property(bool)
    white_box_per_GPU = util.build_typesafe_property(bool)
    EM_FIELD_SCALE_CHANNEL1 = util.build_typesafe_property(EMFieldScaleEnum)
    EM_FIELD_SCALE_CHANNEL2 = util.build_typesafe_property(EMFieldScaleEnum)
    EM_FIELD_SCALE_CHANNEL3 = util.build_typesafe_property(EMFieldScaleEnum)
    preParticleDensCol = util.build_typesafe_property(ColorScaleEnum)
    preChannel1Col = util.build_typesafe_property(ColorScaleEnum)
    preChannel2Col = util.build_typesafe_property(ColorScaleEnum)
    preChannel3Col = util.build_typesafe_property(ColorScaleEnum)
    customNormalizationSI = util.build_typesafe_property(typing.List[float])
    preParticleDens_opacity = util.build_typesafe_property(float)
    preChannel1_opacity = util.build_typesafe_property(float)
    preChannel2_opacity = util.build_typesafe_property(float)
    preChannel3_opacity = util.build_typesafe_property(float)
    preChannel1 = util.build_typesafe_property(str)
    preChannel2 = util.build_typesafe_property(str)
    preChannel3 = util.build_typesafe_property(str)

    _name = "png"

    def __init__(
        self,
        species: Species,
        period: TimeStepSpec,
        axis: str,
        slicePoint: float,
        folder: str,
        scale_image: float,
        scale_to_cellsize: bool,
        white_box_per_GPU: bool,
        EM_FIELD_SCALE_CHANNEL1: EMFieldScaleEnum,
        EM_FIELD_SCALE_CHANNEL2: EMFieldScaleEnum,
        EM_FIELD_SCALE_CHANNEL3: EMFieldScaleEnum,
        preParticleDensCol: ColorScaleEnum,
        preChannel1Col: ColorScaleEnum,
        preChannel2Col: ColorScaleEnum,
        preChannel3Col: ColorScaleEnum,
        customNormalizationSI: typing.List[float],
        preParticleDens_opacity: float,
        preChannel1_opacity: float,
        preChannel2_opacity: float,
        preChannel3_opacity: float,
        preChannel1: str,
        preChannel2: str,
        preChannel3: str,
    ):
        self.species = species
        self.period = period
        self.axis = axis
        self.slicePoint = slicePoint
        self.folder = folder
        self.scale_image = scale_image
        self.scale_to_cellsize = scale_to_cellsize
        self.white_box_per_GPU = white_box_per_GPU
        self.EM_FIELD_SCALE_CHANNEL1 = EM_FIELD_SCALE_CHANNEL1
        self.EM_FIELD_SCALE_CHANNEL2 = EM_FIELD_SCALE_CHANNEL2
        self.EM_FIELD_SCALE_CHANNEL3 = EM_FIELD_SCALE_CHANNEL3
        self.preParticleDensCol = preParticleDensCol
        self.preChannel1Col = preChannel1Col
        self.preChannel2Col = preChannel2Col
        self.preChannel3Col = preChannel3Col
        self.customNormalizationSI = customNormalizationSI
        self.preParticleDens_opacity = preParticleDens_opacity
        self.preChannel1_opacity = preChannel1_opacity
        self.preChannel2_opacity = preChannel2_opacity
        self.preChannel3_opacity = preChannel3_opacity
        self.preChannel1 = preChannel1
        self.preChannel2 = preChannel2
        self.preChannel3 = preChannel3

    def check(self):
        """Validate attributes."""
        try:
            _ = self.species
        except AttributeError:
            raise ValueError("species must be set") from None
        try:
            _ = self.period
        except AttributeError:
            raise ValueError("period must be set") from None
        if self.axis not in ["xy", "yx", "xz", "zx", "yz", "zy"]:
            raise ValueError(f"axis must be 'xy', 'yx', 'xz', 'zx', 'yz', or 'zy', got {self.axis}")
        if self.slicePoint < 0.0 or self.slicePoint > 1.0:
            raise ValueError(f"slicePoint must be in [0, 1], got {self.slicePoint}")
        if self.scale_image <= 0:
            raise ValueError(f"scale_image must be positive, got {self.scale_image}")
        if self.scale_to_cellsize and self.scale_image == 1.0:
            raise ValueError(f"scale_image must not be 1.0 when scale_to_cellsize is True, got {self.scale_image}")
        if self.preParticleDens_opacity < 0 or self.preParticleDens_opacity > 1:
            raise ValueError(f"preParticleDens_opacity must be in [0, 1], got {self.preParticleDens_opacity}")
        if self.preChannel1_opacity < 0 or self.preChannel1_opacity > 1:
            raise ValueError(f"preChannel1_opacity must be in [0, 1], got {self.preChannel1_opacity}")
        if self.preChannel2_opacity < 0 or self.preChannel2_opacity > 1:
            raise ValueError(f"preChannel2_opacity must be in [0, 1], got {self.preChannel2_opacity}")
        if self.preChannel3_opacity < 0 or self.preChannel3_opacity > 1:
            raise ValueError(f"preChannel3_opacity must be in [0, 1], got {self.preChannel3_opacity}")
        for channel, name in [
            (self.preChannel1, "preChannel1"),
            (self.preChannel2, "preChannel2"),
            (self.preChannel3, "preChannel3"),
        ]:
            if not isinstance(channel, str) or not channel.strip():
                raise ValueError(f"{name} must be a non-empty string, got {channel}")
        if len(self.customNormalizationSI) != 3:
            raise ValueError(
                f"customNormalizationSI must contain exactly 3 floats, got {len(self.customNormalizationSI)}"
            )
        for val in self.customNormalizationSI:
            if not isinstance(val, float):
                raise ValueError(f"customNormalizationSI values must be floats, got {val}")
        if not isinstance(self.EM_FIELD_SCALE_CHANNEL1, EMFieldScaleEnum):
            raise ValueError(
                f"EM_FIELD_SCALE_CHANNEL1 must be in {list(EMFieldScaleEnum)}, got {self.EM_FIELD_SCALE_CHANNEL1}"
            )
        if not isinstance(self.EM_FIELD_SCALE_CHANNEL2, EMFieldScaleEnum):
            raise ValueError(
                f"EM_FIELD_SCALE_CHANNEL2 must be in {list(EMFieldScaleEnum)}, got {self.EM_FIELD_SCALE_CHANNEL2}"
            )
        if not isinstance(self.EM_FIELD_SCALE_CHANNEL3, EMFieldScaleEnum):
            raise ValueError(
                f"EM_FIELD_SCALE_CHANNEL3 must be in {list(EMFieldScaleEnum)}, got {self.EM_FIELD_SCALE_CHANNEL3}"
            )
        if not isinstance(self.preParticleDensCol, ColorScaleEnum):
            raise ValueError(f"preParticleDensCol must be in {list(ColorScaleEnum)}, got {self.preParticleDensCol}")
        if not isinstance(self.preChannel1Col, ColorScaleEnum):
            raise ValueError(f"preChannel1Col must be in {list(ColorScaleEnum)}, got {self.preChannel1Col}")
        if not isinstance(self.preChannel2Col, ColorScaleEnum):
            raise ValueError(f"preChannel2Col must be in {list(ColorScaleEnum)}, got {self.preChannel2Col}")
        if not isinstance(self.preChannel3Col, ColorScaleEnum):
            raise ValueError(f"preChannel3Col must be in {list(ColorScaleEnum)}, got {self.preChannel3Col}")

    def _get_serialized(self) -> typing.Dict:
        """Return the serialized representation of the object."""
        self.check()
        custom_normalization_si_serialized = [{"value": val} for val in self.customNormalizationSI]
        return {
            "species": self.species.get_rendering_context(),
            "period": self.period.get_rendering_context(),
            "axis": self.axis,
            "slicePoint": self.slicePoint,
            "folder": self.folder,
            "scale_image": self.scale_image,
            "scale_to_cellsize": self.scale_to_cellsize,
            "white_box_per_GPU": self.white_box_per_GPU,
            "EM_FIELD_SCALE_CHANNEL1": self.EM_FIELD_SCALE_CHANNEL1.value,
            "EM_FIELD_SCALE_CHANNEL2": self.EM_FIELD_SCALE_CHANNEL2.value,
            "EM_FIELD_SCALE_CHANNEL3": self.EM_FIELD_SCALE_CHANNEL3.value,
            "preParticleDensCol": self.preParticleDensCol.value,
            "preChannel1Col": self.preChannel1Col.value,
            "preChannel2Col": self.preChannel2Col.value,
            "preChannel3Col": self.preChannel3Col.value,
            "customNormalizationSI": custom_normalization_si_serialized,
            "preParticleDens_opacity": self.preParticleDens_opacity,
            "preChannel1_opacity": self.preChannel1_opacity,
            "preChannel2_opacity": self.preChannel2_opacity,
            "preChannel3_opacity": self.preChannel3_opacity,
            "preChannel1": self.preChannel1,
            "preChannel2": self.preChannel2,
            "preChannel3": self.preChannel3,
        }
