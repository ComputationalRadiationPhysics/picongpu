"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import logging
import math
from typing import Annotated

import numpy as np
from pydantic import BeforeValidator

from .. import constants

PositiveFloat = Annotated[
    float,
    BeforeValidator(lambda v: float(v) if (float(v) > 0) else (_ for _ in ()).throw(ValueError("value must be > 0"))),
]
"""float that must be strictly > 0"""


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

        factor = constants.m_e * constants.c**2 * k0 / constants.q_e
        if (E0 is not None) and (a0 is not None):
            # Both amplitudes are set. This happens when an already-validated laser
            # instance is re-validated (e.g. when it is passed to
            # ``Simulation.add_laser``): at construction only one amplitude was given
            # and the other was derived. Treat that as "already resolved" and keep the
            # values, checking them for consistency rather than rejecting them. This is
            # what makes the validator idempotent, as required now that picmistandard
            # types ``Simulation.lasers`` and re-validates the instances it stores.
            if not math.isclose(E0, a0 * factor, rel_tol=1.0e-6):
                raise ValueError(f"Inconsistent E0 and a0: {E0=} and {a0=}.")
            return a0, E0

        if E0 is None:
            E0 = a0 * factor
        if a0 is None:
            a0 = E0 / factor
        return a0, E0

    def _pulse_duration_sigma_si(self):
        """Pulse duration in s, as the 1 sigma of a standard gaussian for the intensity (E^2).

        This is the quantity PIConGPU's laser parameters use (``PULSE_DURATION_SI``
        in ``incidentField.param``). Classes whose PICMI ``duration`` has different
        semantics (e.g. the PICMI standard's 1/e field width, see GaussianLaser)
        must override this to return the converted value.
        """
        return self.duration

    def _entry_axis(self) -> int:
        """Axis index (0/1/2) of the dominant propagation-direction component.

        The laser enters the simulation box through the coordinate face whose normal
        is this axis: a positive dominant component means entry from the ``Min`` face
        (low side) and a negative one from the ``Max`` face (high side).
        """
        direction = np.asarray(self.propagation_direction, dtype=float)
        axis = int(np.argmax(np.abs(direction)))
        if direction[axis] == 0.0:
            raise ValueError(
                "The laser propagation direction has no dominant (largest-magnitude) "
                "component, so the face through which it enters the simulation box "
                f"cannot be determined. You gave {self.propagation_direction=}."
            )
        return axis

    def _entry_face(self) -> str:
        """Name of the coordinate face the laser enters through (e.g. ``YMin``)."""
        axis = self._entry_axis()
        suffix = "Min" if self.propagation_direction[axis] > 0 else "Max"
        return "xyz"[axis].upper() + suffix

    def _compute_pulse_init(self):
        pulse_init = (
            -2.0
            * self.centroid_position[1]
            / (self.propagation_direction[1] * constants.c)
            / self._pulse_duration_sigma_si()
        )  # unit: multiple of the laser pulse duration (1 sigma of the intensity)
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

        axis = self._entry_axis()
        entry_axis_name = "xyz"[axis]
        if self.centroid_position[axis] * self.propagation_direction[axis] >= 0.0:
            raise ValueError(
                "The laser maximum (centroid) must be located outside of the "
                "simulation box on the entry side, otherwise it is impossible to "
                "correctly initialize it using a huygens surface in the box. The "
                f"laser enters through the {self._entry_face()} face, so the "
                f"{entry_axis_name}-component of the centroid must point against "
                f"the propagation direction (centroid_{entry_axis_name} * "
                f"direction_{entry_axis_name} < 0). You gave "
                f"{self.centroid_position=} and {self.propagation_direction=}."
            )
