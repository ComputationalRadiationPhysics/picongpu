"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import typeguard
from typing import Optional


@typeguard.typechecked
class BoundElectronDensity:
    """
    Represents the BoundElectronDensity data source for the openPMD plugin in PIConGPU.

    This class defines the bound electron density field, derived from particle species at runtime,
    which can be output using the openPMD plugin. This is typically used for partially ionized ions.
    An optional filter can be applied to select which particles contribute to the bound electron density.

    Parameters
    ----------
    filter: str, optional
        Name of a deterministic filter to apply to particles, as defined in particleFilters.param
        (see picongpu/include/picongpu/param/particleFilters.param).
        The default filter is "all" (selects all valid particles). Additional filters, such as
        "relativeGlobalDomainPosition" (selects particles in a global domain range), can be defined
        in your local particleFilters.param file. If None, no filter is applied. Valid filters must be
        deterministic and are listed in the PIConGPU command-line help for --openPMD.source.
    """

    def __init__(self, filter: Optional[str] = None):
        self.filter = filter
        self.check()

    def check(self):
        """
        Validate the provided filter.
        """
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

    def get_source_string(self) -> str:
        """
        Return the source string for use in --openPMD.source.

        Returns
        -------
        str
            The dataset name with optional filter (e.g., "bound_electron_density" or
            "bound_electron_density:filterX").
        """
        if self.filter:
            return f"bound_electron_density:{self.filter}"
        return "bound_electron_density"

    def get_as_pypicongpu(self) -> str:
        """
        Return the source string for PyPIConGPU integration.

        Returns
        -------
        str
            The dataset name with optional filter.
        """
        return self.get_source_string()
