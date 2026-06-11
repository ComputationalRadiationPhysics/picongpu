"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

import logging

import numpy as np

from .. import constants


def scalarProduct(a: list[float], b: list[float]) -> float:
    return np.dot(a, b).tolist()


def crossProduct(a: list[float], b: list[float]) -> list[float]:
    return np.cross(a, b).tolist()


def difference(a: list[float], b: list[float]) -> list[float]:
    return (np.asarray(a) - b).tolist()


class BaseLaser:
    """
    Base class for all PICMI laser implementations to reduce code duplication
    """

    def _propagation_connects_centroid_and_focus(self):
        # check that propagation_direction is parallel to the difference of focal_position and centroid_position
        diff_vec = difference(self.focal_position, self.centroid_position)
        cross_vec = crossProduct(diff_vec, self.propagation_direction)
        length_of_cross_product = scalarProduct(cross_vec, cross_vec)
        return length_of_cross_product < 1.0e-5

    def _compute_E0_and_a0(self, k0, E0, a0):
        if (E0 is None) and (a0 is None):
            raise ValueError("Both E0 or a0 are None. You must specify exactly one.")
        if (E0 is not None) and (a0 is not None):
            raise ValueError("Only one of E0 or a0 should be specified. You set both.")

        if E0 is None:
            E0 = a0 * constants.m_e * constants.c**2 * k0 / constants.q_e
        if a0 is None:
            a0 = E0 / (constants.m_e * constants.c**2 * k0 / constants.q_e)
        return a0, E0

    def _compute_pulse_init(self):
        pulse_init = (
            -2.0 * self.centroid_position[1] / (self.propagation_direction[1] * constants.c) / self.duration
        )  # unit: multiple of laser pulse duration
        # @todo extend this to other propagation directions than +y
        if pulse_init < 3.0:
            logging.warning(
                "set centroid_position and propagation_direction indicate that laser "
                + "initalization might be too short.\n"
                + f"Details: {pulse_init=} < 3"
            )
        return pulse_init

    def _validate_common_properties(self):
        """Common validation logic for all lasers"""

        if not np.allclose(n := np.linalg.norm(self.polarization_direction), 1):
            raise ValueError(
                "The polarization direction vector must be normalized. "
                f"You gave {self.polarization_direction=} with norm {n}."
            )

        if not np.allclose(n := np.linalg.norm(self.propagation_direction), 1):
            raise ValueError(
                "The propagation direction vector must be normalized. "
                f"You gave {self.propagation_direction=} with norm {n}."
            )

        if scalarProduct(self.propagation_direction, [0.0, 1.0, 0.0]) <= 0.0:
            raise ValueError(
                "Laser propagation parallel to the y-plane or pointing outside "
                "from the inside of the simulation box is not supported by this "
                f"laser in PICMI. You gave {self.propagation_direction=}."
            )

        if self.centroid_position[1] > 0:
            raise ValueError(
                "The laser maximum must be outside of the "
                "simulation box, otherwise it is impossible to correctly initialize"
                "it using a huygens surface in the box, centroid_y <= 0. "
                f"You gave {self.centroid_position=}."
            )
