"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from typeguard import typechecked
from .... import util
from ....rendering import RenderedObject

# Note to the future maintainer:
# If you want to add another way to specify the temperature, please turn
# Temperature() into an (abstract) parent class, and add one child class per
# method. (Currently only initialization by giving a temperature in keV is
# supported, so such a structure would be overkill.)


@typechecked
class Temperature(RenderedObject):
    """
    Initialize momentum from temperature
    """

    temperature_kev = util.build_typesafe_property(float)
    """temperature to use in keV"""

    def check(self) -> None:
        """
        check validity of self

        pass silently if okay, raise on error
        """
        if self.temperature_kev <= 0:
            raise ValueError("temperature must be >0")

    def _get_serialized(self) -> dict:
        self.check()
        return {
            "temperature_kev": self.temperature_kev,
        }
