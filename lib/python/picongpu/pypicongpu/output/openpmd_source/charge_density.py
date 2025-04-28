"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .. import util
import typeguard
import typing


@typeguard.typechecked
class ChargeDensity:
    filter = util.build_typesafe_property(typing.Optional[str])

    def __init__(self, filter: typing.Optional[str] = None):
        self.filter = filter
        self.check()

    def check(self):
        if self.filter is not None and not isinstance(self.filter, str):
            raise ValueError(
                f"Filter must be a string or None, got {type(self.filter)}. "
                "Valid filter names are defined in particleFilters.param "
                "(see picongpu/include/picongpu/param/particleFilters.param). "
                "The default filter is 'all' (selects all valid particles). Additional filters, such as "
                "'relativeGlobalDomainPosition' (selects particles in a global domain range), can be defined "
                "in your local particleFilters.param file. Valid filters are listed in the PIConGPU "
                "command-line help for --openPMD.source."
            )

    def _get_serialized(self) -> typing.Dict:
        return {
            "dataset": "charge_density",
            "filter": self.filter,
        }
