#!/usr/bin/env python
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Places three Gaussian lasers on three different entry faces: one entering
through the low-x face (propagation along +x), one through the high-y face
(propagation along -y) and one through the low-z face (propagation along +z).
The generated ``incidentField.param`` is then inspected to show that every
laser was placed on the face matching its propagation direction.
"""

# BEGIN-LASER-ENTRY-FACES
import re
from pathlib import Path

from picongpu import picmi

NUM_CELLS = [192, 2048, 192]
CELL_SIZE = [0.1772e-6, 0.4430e-7, 0.1772e-6]
CENTER = [NUM_CELLS[i] * CELL_SIZE[i] / 2.0 for i in range(3)]


def laser(propagation_direction, polarization_direction, centroid_position):
    return picmi.GaussianLaser(
        wavelength=0.8e-6,
        waist=5.0e-6,
        duration=5.0e-15,
        # a propagation direction pointing into the box; the entry face is the
        # coordinate face whose normal is the dominant component of this vector:
        propagation_direction=propagation_direction,
        polarization_direction=polarization_direction,
        focal_position=CENTER,
        # the centroid at time zero lies outside the box on the entry side:
        centroid_position=centroid_position,
        a0=8.0,
    )


lasers = [
    # enters through the low-x face, propagates along +x:
    laser([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-5.0e-6, CENTER[1], CENTER[2]]),
    # enters through the high-y face, propagates along -y:
    laser([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [CENTER[0], 1.5 * NUM_CELLS[1] * CELL_SIZE[1], CENTER[2]]),
    # enters through the low-z face, propagates along +z:
    laser([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [CENTER[0], CENTER[1], -5.0e-6]),
]

grid = picmi.Cartesian3DGrid(
    number_of_cells=NUM_CELLS,
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[NUM_CELLS[i] * CELL_SIZE[i] for i in range(3)],
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid)
simulation = picmi.Simulation(max_steps=100, solver=solver, lasers=lasers)
simulation.write_input_file(Path("laser_entry_faces_setup"))


def entry_face_of(incident_field, index):
    """Return the face whose MakeSeq contains LaserProfile_<index>."""
    for face in ("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax"):
        block = re.search(rf"using {face} = MakeSeq_t<(.*?)>;", incident_field, re.DOTALL).group(1)
        if f"LaserProfile_{index}" in block:
            return face
    return None


incident_field = Path("laser_entry_faces_setup/include/picongpu/param/incidentField.param").read_text()
for index, expected in enumerate(["XMin", "YMax", "ZMin"]):
    face = entry_face_of(incident_field, index)
    assert face == expected, f"laser {index} was placed on {face}, expected {expected}"
    print(f"laser {index} enters through {face}")
print("It worked!")
# END-LASER-ENTRY-FACES
