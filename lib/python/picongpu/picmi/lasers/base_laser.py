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

        The dominant component is ``argmax(abs(direction))``, which resolves exact
        magnitude ties to the *first* axis in x, y, z order (no tolerance). A
        genuinely-ambiguous 45° diagonal is therefore assigned to the lowest-index
        tied axis by convention, not by physics; near-ties are decided by the larger
        component and are unambiguous.
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

    def _focus_position_si(self) -> np.ndarray:
        """Focus position in SI, as used by the C++ ``FOCUS_POSITION_*_SI``.

        The C++ profile projects the focus onto the generation surface to find the
        surface origin. Most lasers carry a ``focal_position``; the plane wave has
        its focus fixed at the coordinate origin (and exposes only ``focus_pos``).
        """
        focal_position = getattr(self, "focal_position", None)
        if focal_position is None:
            return np.zeros(3, dtype=float)
        return np.asarray(focal_position, dtype=float)

    def _huygens_surface_origin_si(self, cell_size, domain_cells) -> np.ndarray:
        """Port of ``incidentField::detail::BaseFunctorE::getOrigin()`` in SI.

        The origin is the intersection of the line through the focus along the
        propagation direction with the generation (Huygens) surface, choosing the
        point the laser encounters first. The generation surface is displaced by
        0.75 cells inwards from the configured ``POSITION`` indices. This is the
        time reference the C++ profile evaluates against; it is needed only by
        the translation layer, never by the user-facing analytic formula.

        When no grid is known (standalone translation of an unbound laser), the
        surface is taken to sit at the coordinate origin, i.e. the historical
        origin-at-zero convention. A real simulation always supplies the grid.
        """
        direction = np.asarray(self.propagation_direction, dtype=float)
        if cell_size is None or domain_cells is None:
            return np.zeros(3, dtype=float)
        focus = self._focus_position_si()
        cell_size = np.asarray(cell_size, dtype=float)
        domain_cells = np.asarray(domain_cells, dtype=float)
        positions = self.picongpu_huygens_surface_positions
        origin_p = -np.inf
        for axis in range(len(cell_size)):
            if abs(direction[axis]) <= np.finfo(float).eps:
                continue
            min_position = (positions[axis][0] + 0.75) * cell_size[axis]
            max_index = positions[axis][1] if positions[axis][1] > 0 else domain_cells[axis] + positions[axis][1]
            max_position = (max_index - 0.75) * cell_size[axis]
            axis_p = min(
                (min_position - focus[axis]) / direction[axis],
                (max_position - focus[axis]) / direction[axis],
            )
            origin_p = max(origin_p, axis_p)
        return focus + origin_p * direction

    def _compute_pulse_init(self, cell_size=None, domain_cells=None):
        """``pulse_init`` (in units of the pulse duration) used by pypicongpu.

        The frontend exposes the PICMI-standard, centroid-based definition only:
        the pulse maximum is at ``centroid_position`` at ``t = 0``. The core,
        however, times its profile against the actual displaced Huygens-surface
        origin. This conversion (= the translation layer PICMI -> pypicongpu)
        re-expresses the user's centroid in the core's reference frame:

            pulse_init = 2 * dot(origin - centroid, direction) / (c * sigma)

        For propagation along +y and an origin at the coordinate origin this
        reduces to the historical ``-2 * centroid_y / (c * sigma)``.
        """
        distance = scalarProduct(
            difference(self._huygens_surface_origin_si(cell_size, domain_cells), self.centroid_position),
            self.propagation_direction,
        )
        pulse_init = 2.0 * distance / constants.c / self._pulse_duration_sigma_si()
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
        if self.centroid_position[axis] * self.propagation_direction[axis] > 0.0:
            raise ValueError(
                "The laser maximum (centroid) must be located outside of the "
                "simulation box on the entry side, otherwise it is impossible to "
                "correctly initialize it using a huygens surface in the box. The "
                f"laser enters through the {self._entry_face()} face, so the "
                f"{entry_axis_name}-component of the centroid must not point along "
                f"the propagation direction (centroid_{entry_axis_name} * "
                f"direction_{entry_axis_name} <= 0). You gave "
                f"{self.centroid_position=} and {self.propagation_direction=}."
            )
