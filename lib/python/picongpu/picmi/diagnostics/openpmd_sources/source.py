"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .bound_electron_density import BoundElectronDensity
from .charge_density import ChargeDensity
from .counter import Counter
from .density import Density
from .energy import Energy
from .energy_density import EnergyDensity
from .energy_density_cutoff import EnergyDensityCutoff
from .larmor_power import LarmorPower
from .macro_counter import MacroCounter
from .mid_current_density_component import MidCurrentDensityComponent
from .momentum import Momentum
from .momentum_density import MomentumDensity
from .weighted_velocity import WeightedVelocity

import typeguard
from typing import List, Union


@typeguard.typechecked
class Source:
    """
    Consolidates data sources for the openPMD plugin in PIConGPU.

    This class aggregates individual data sources (e.g., ChargeDensity) or predefined
    keywords (species_all, fields_all) to define the --openPMD.source parameter:
    https://picongpu.readthedocs.io/en/latest/usage/plugins/openPMD.html

    Parameters
    ----------
    sources: List[Union[str, <DataSourceClasses>]]
        List of data sources, either as strings (e.g., "species_all", "fields_all") or
        as data source objects (e.g., ChargeDensity, Density).
    """

    # List of valid data source classes
    VALID_SOURCE_CLASSES = (
        BoundElectronDensity,
        ChargeDensity,
        Counter,
        Density,
        Energy,
        EnergyDensity,
        EnergyDensityCutoff,
        LarmorPower,
        MacroCounter,
        MidCurrentDensityComponent,
        Momentum,
        MomentumDensity,
        WeightedVelocity,
    )

    def __init__(self, sources: List[Union[str, *VALID_SOURCE_CLASSES]]):
        self.sources = sources
        self.check()

    def check(self):
        """
        Validate the provided sources.
        """
        valid_strings = {"species_all", "fields_all"}
        for src in self.sources:
            if isinstance(src, str):
                if src not in valid_strings:
                    raise ValueError(f"Invalid source string: '{src}'. Must be one of {valid_strings}.")
            elif not isinstance(src, self.VALID_SOURCE_CLASSES):
                raise ValueError(
                    f"Invalid source type: {type(src)}. Must be str or one of {self.VALID_SOURCE_CLASSES}."
                )

    def get_as_pypicongpu(self) -> List[str]:
        """
        Convert sources to a list of strings for PyPIConGPU integration.

        Returns
        -------
        List[str]
            List of source strings (e.g., ["species_all", "charge_density:filterX"]).
        """
        return [src if isinstance(src, str) else src.get_as_pypicongpu() for src in self.sources]
