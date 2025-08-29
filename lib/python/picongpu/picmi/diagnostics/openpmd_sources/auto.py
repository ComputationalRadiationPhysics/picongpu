"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .source_base import SourceBase
from picongpu.pypicongpu.output.openpmd_sources import Auto as PyPIConGPUAuto
import typeguard
import typing


@typeguard.typechecked
class Auto(SourceBase):
    """
    Default data source for openPMD output in PIConGPU.

    Provides a convenient way to dump default simulation data (e.g., all particle species and fields)
    using the openPMD standard, with defaults determined by the PIC code.
    """

    def __init__(self, filter: str = "species_all"):
        self._filter = filter
        self.check()

    @property
    def filter(self) -> str:
        return self._filter

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self._filter, str):
            raise TypeError(f"Filter must be a string, got {type(self._filter)}")
        if self._filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self._filter}")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: typing.Optional[typing.Dict] = None,
        time_step_size: float = 0.0,
        num_steps: int = 0,
        simulation_box=None,
    ) -> PyPIConGPUAuto:
        self.check()
        return PyPIConGPUAuto(filter=self._filter)
