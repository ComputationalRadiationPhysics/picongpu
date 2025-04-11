"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Richard Pausch, Julian Lenz
License: GPLv3+
"""

from .timestepspec import TimeStepSpec
from .. import util
from .plugin import Plugin

import typeguard


@typeguard.typechecked
class Auto(Plugin):
    """
    Class to provide output **without further configuration**.

    This class requires a period (in time steps) and will enable as many output
    plugins as feasable for all species.
    Note: The list of species from the initmanager is used during rendering.

    No further configuration is possible!
    If you want to provide additional configuration for plugins,
    create a separate class.
    """

    period = util.build_typesafe_property(TimeStepSpec)
    """period to print data at"""

    _name = "auto"

    def __init__(self):
        pass

    def check(self) -> None:
        """
        validate attributes
        """
        pass

    def _get_serialized(self) -> dict:
        self.check()
        return {
            "period": self.period.get_rendering_context(),
            # helper to avoid repeating code
            "png_axis": [
                {"axis": "yx"},
                {"axis": "yz"},
            ],
        }
