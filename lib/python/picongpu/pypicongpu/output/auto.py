"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch
License: GPLv3+
"""

from .. import util
from ..rendering import RenderedObject
from typeguard import typechecked


@typechecked
class Auto(RenderedObject):
    """
    Class to provide output **without further configuration**.

    This class requires a period (in time steps) and will enable as many output
    plugins as feasable for all species.
    Note: The list of species from the initmanager is used during rendering.

    No further configuration is possible!
    If you want to provide additional configuration for plugins,
    create a separate class.
    """

    period = util.build_typesafe_property(int)
    """period to print data at"""

    def check(self) -> None:
        """
        validate attributes

        if ok pass silently, otherwise raises error

        :raises ValueError: period is non-negative integer
        :raises ValueError: species_names contains empty string
        :raises ValueError: species_names contains non-unique name
        """
        if 1 > self.period:
            raise ValueError("period must be non-negative integer")

    def _get_serialized(self) -> dict:
        self.check()
        return {
            "period": self.period,

            # helper to avoid repeating code
            "png_axis": [
                {"axis": "yx"},
                {"axis": "yz"},
            ],
        }
