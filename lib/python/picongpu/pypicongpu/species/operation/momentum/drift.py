"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from ....rendering import RenderedObject
from .... import util

import typeguard
import typing
import math
from scipy import constants

# Note to the future maintainer:
# If you want to add another way to specify the drift, please turn
# Drift() into an (abstract) parent class, and add one child class per
# method.


@typeguard.typechecked
class Drift(RenderedObject):
    """
    Add drift to a species (momentum)

    Note that the drift is specified by a direction (normalized velocity
    vector) and gamma. Helpers to load from other representations (originating
    from PICMI) are provided.
    """

    direction_normalized = util.build_typesafe_property(typing.Tuple[float, float, float])
    """direction of drift, length of one"""

    gamma = util.build_typesafe_property(float)
    """gamma, the physicists know"""

    def __check_vector_real(self, vector: typing.Tuple[float, float, float]) -> None:
        """
        check that a vector only contains real components

        passes silently if OK, throws otherwise

        :param vector: three-tuple to check
        """
        for invalid in [math.inf, -math.inf, math.nan]:
            if invalid in vector:
                raise ValueError(
                    "vector may only contain real components, offending axis: {}".format(
                        ["x", "y", "z"][vector.index(invalid)]
                    )
                )

    def check(self) -> None:
        """
        check attributes for correctness

        pass silently if everything is OK,
        throw error otherwise
        """
        invalids = [math.inf, -math.inf, math.nan]
        if self.gamma in invalids:
            raise ValueError("gamma must be real")
        if self.gamma < 1:
            raise ValueError("gamma must be >=1")

        self.__check_vector_real(self.direction_normalized)

        vector_length = sum(map(lambda n: n**2, self.direction_normalized))
        if 1 != round(vector_length, 6):
            raise ValueError("direction must be normalized (current length: {})".format(vector_length))

    def fill_from_velocity(self, velocity: typing.Tuple[float, float, float]) -> None:
        """
        set attributes to represent given velocity vector

        computes gamma and direction_normalized for self

        :param velocity: velocity given as vector
        """
        self.__check_vector_real(velocity)
        if (0, 0, 0) == velocity:
            raise ValueError("velocity must not be zero")

        velocity_linear = math.sqrt(sum(map(lambda x: x**2, velocity)))
        if velocity_linear >= constants.speed_of_light:
            raise ValueError(
                "linear velocity must be less than the speed of light (currently: {})".format(velocity_linear)
            )

        gamma = math.sqrt(1 / (1 - (velocity_linear**2 / constants.speed_of_light**2)))

        self.direction_normalized = tuple(map(lambda x: x / velocity_linear, velocity))
        self.gamma = gamma

    def fill_from_gamma_velocity(self, gamma_velocity: typing.Tuple[float, float, float]) -> None:
        """
        set attributes to represent given velocity vector multiplied with gamma

        computes gamma and direction_normalized for self

        :param velocity: velocity given as vector multiplied with gamma
        """
        self.__check_vector_real(gamma_velocity)
        if (0, 0, 0) == gamma_velocity:
            raise ValueError("velocity must not be zero")

        gamma_velocity_linear = math.sqrt(sum(map(lambda x: x**2, gamma_velocity)))
        gamma = math.sqrt(1 + ((gamma_velocity_linear) ** 2 / constants.speed_of_light**2))

        self.direction_normalized = tuple(map(lambda x: x / gamma_velocity_linear, gamma_velocity))
        self.gamma = gamma

    def _get_serialized(self) -> dict:
        return {
            "gamma": self.gamma,
            "direction_normalized": {
                "x": self.direction_normalized[0],
                "y": self.direction_normalized[1],
                "z": self.direction_normalized[2],
            },
        }
