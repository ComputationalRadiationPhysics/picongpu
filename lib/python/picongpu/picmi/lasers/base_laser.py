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
from pydantic import BeforeValidator, computed_field

from ...pypicongpu import laser as pypicongpu_laser
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

    # Explicit per-laser selection of the Huygens faces this pulse is injected
    # through. If left as None, all faces crossed by the propagation direction
    # are used (see _entry_faces). Declared here rather than on each concrete
    # laser because _entry_faces reads it and every standard laser rendered via
    # the on_* membership flags supports it.
    picongpu_entry_faces: list[str] | None = None

    def _uses_entry_faces(self) -> bool:
        """Whether the entry-face selection actually drives this laser's placement.

        Standard lasers are rendered under their ``on_*`` face guards and their
        derived/explicit face list is validated by :meth:`validate_entry_faces`.
        TWTS keeps a dedicated fixed placement (always ``YMin`` plus
        ``ZMin``/``ZMax`` chosen by the incidence-angle sign) that the template
        selects via ``type_twts``, so its derived face list is unused and must
        not be validated against the grid dimensionality.
        """
        return True

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

    def _entry_faces(self) -> list[str]:
        """Coordinate faces this laser is injected through.

        If the user gave the PIConGPU extension keyword
        ``picongpu_entry_faces`` it is used verbatim (explicit per-laser list,
        answer 1 of https://github.com/chillenzer-agents/picongpu/issues/180),
        after checking the names and rejecting duplicates. Otherwise the default
        is *all crossed faces*: for every non-zero propagation-direction
        component ``d > 0`` the ``d``-Min face and for ``d < 0`` the ``d``-Max
        face (answer 2). "Crossed" therefore means that the propagation direction
        has a non-zero component pointing inward through that boundary. For the
        oblique direction ``(0.5, 0, sqrt(3)/2)`` this is ``["XMin", "ZMin"]``.

        Z faces are not dropped here: a 2D simulation rejects a Z selection via
        :meth:`validate_entry_faces` (answer 5), which is called by
        :class:`~picongpu.picmi.Simulation` where the dimensionality is known.
        """
        if (explicit := self.picongpu_entry_faces) is not None:
            faces = list(explicit)
            illegal = [face for face in faces if face not in pypicongpu_laser.ENTRY_FACES]
            if illegal:
                raise ValueError(
                    f"Unknown Huygens entry face(s) {illegal}. "
                    f"Valid faces are {list(pypicongpu_laser.ENTRY_FACES)}. You gave {explicit=}."
                )
            if len(set(faces)) != len(faces):
                raise ValueError(f"Duplicate Huygens entry faces are not allowed. You gave {explicit=}.")
            if not faces:
                raise ValueError("At least one Huygens entry face must be selected.")
        else:
            faces = pypicongpu_laser.entry_faces_from_direction(self.propagation_direction)
        return faces

    def validate_entry_faces(self, dimension: int):
        """Reject entry-face selections that are impossible in ``dimension`` dimensions.

        A 2D (2D3V) simulation has no ``z`` coordinate and therefore no Z face to
        inject through, so any selection containing ``ZMin``/``ZMax`` is invalid
        (answer 5 of https://github.com/chillenzer-agents/picongpu/issues/180).
        This is called by :class:`~picongpu.picmi.Simulation`, which knows the
        grid's dimensionality during translation.
        """
        if dimension < 3 and self._uses_entry_faces():
            faces = self._entry_faces()
            z_faces = [face for face in faces if face.startswith("Z")]
            if z_faces:
                raise ValueError(
                    "Injection through Z faces is not possible in a 2D simulation (there is no z coordinate). "
                    f"The selected entry faces {faces} include {z_faces}. "
                    "Choose a propagation direction without a z-component or set `picongpu_entry_faces` "
                    "to a face subset that avoids ZMin/ZMax."
                )

    @computed_field
    @property
    def entry_faces(self) -> list[str]:
        """The face list handed to the PyPIConGPU laser (see :meth:`_entry_faces`)."""
        return self._entry_faces()

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

        # The centroid must lie outside the box on every entry side, so that the
        # pulse has not yet reached the Huygens surface at t=0. For each selected
        # face the corresponding centroid component must not point along the
        # propagation direction (centroid_d * direction_d <= 0).
        for face in self._entry_faces():
            axis = "XYZ".index(face[0])
            entry_axis_name = "xyz"[axis]
            if self.centroid_position[axis] * self.propagation_direction[axis] > 0.0:
                raise ValueError(
                    "The laser maximum (centroid) must be located outside of the "
                    "simulation box on each entry side, otherwise it is impossible to "
                    "correctly initialize it using a huygens surface in the box. The "
                    f"laser enters through the {face} face, so the "
                    f"{entry_axis_name}-component of the centroid must not point along "
                    f"the propagation direction (centroid_{entry_axis_name} * "
                    f"direction_{entry_axis_name} <= 0). You gave "
                    f"{self.centroid_position=} and {self.propagation_direction=}."
                )
