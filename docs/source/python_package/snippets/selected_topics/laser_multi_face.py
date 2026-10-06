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

Injects a single obliquely incident pulse through *all* Huygens faces its
propagation direction crosses. The direction ``(0.5, 0, sqrt(3)/2)`` crosses
both the ``XMin`` and the ``ZMin`` face; the default behaviour is to drive both
of them with the same pulse profile, so one physical wavefront enters the box
across both faces. The generated ``incidentField.param`` is inspected to show
that the same profile is listed under both face guards, and then an explicit
per-laser face list is used to select a strict subset.
"""

# BEGIN-LASER-MULTI-FACE
import re
from pathlib import Path

from picongpu import picmi

NUM_CELLS = [192, 256, 192]
CELL_SIZE = [0.1772e-6, 0.4430e-7, 0.1772e-6]
LOWER = [0.0, 0.0, 0.0]
UPPER = [n * s for n, s in zip(NUM_CELLS, CELL_SIZE)]

# an oblique direction with non-zero x- and z-components:
# it crosses the low-x face (XMin) and the low-z face (ZMin)
PROPAGATION = [0.5, 0.0, 0.8660254037844386]


def oblique_laser(**kwargs):
    return picmi.GaussianLaser(
        wavelength=0.8e-6,
        waist=5.0e-6,
        duration=5.0e-15,
        propagation_direction=PROPAGATION,
        polarization_direction=[0.0, 1.0, 0.0],
        focal_position=[0.0, 0.0, 0.0],
        # the centroid at time zero must lie outside the box on *every* entry
        # side, i.e. along the negative propagation direction here
        centroid_position=[-2.5e-6, 0.0, -4.330127018922193e-6],
        a0=8.0,
        **kwargs,
    )


grid = picmi.Cartesian3DGrid(
    number_of_cells=NUM_CELLS,
    lower_bound=LOWER,
    upper_bound=UPPER,
    lower_boundary_conditions=["open", "open", "open"],
    upper_boundary_conditions=["open", "open", "open"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.95, grid=grid)

# default: both crossed faces (XMin and ZMin) carry the one pulse
simulation = picmi.Simulation(max_steps=100, solver=solver, lasers=[oblique_laser()])
simulation.write_input_file(Path("laser_multi_face_setup"))


def faces_of(incident_field, index):
    """Return the faces whose MakeSeq contains LaserProfile_<index>."""
    return [
        face
        for face in ("XMin", "XMax", "YMin", "YMax", "ZMin", "ZMax")
        if f"LaserProfile_{index}"
        in re.search(rf"using {face} = MakeSeq_t<(.*?)>;", incident_field, re.DOTALL).group(1)
    ]


incident_field = Path("laser_multi_face_setup/include/picongpu/param/incidentField.param").read_text()
faces = faces_of(incident_field, 0)
assert faces == ["XMin", "ZMin"], f"expected XMin and ZMin, got {faces}"
print(f"default: one pulse injected through {faces}")

# explicit per-laser selection may inject on a strict subset of the crossed
# faces (here: only XMin)
subset_simulation = picmi.Simulation(
    max_steps=100,
    solver=solver,
    lasers=[oblique_laser(picongpu_entry_faces=["XMin"])],
)
subset_simulation.write_input_file(Path("laser_multi_face_subset_setup"))
subset_incident_field = Path("laser_multi_face_subset_setup/include/picongpu/param/incidentField.param").read_text()
subset_faces = faces_of(subset_incident_field, 0)
assert subset_faces == ["XMin"], f"expected only XMin, got {subset_faces}"
print(f"override: one pulse injected through {subset_faces}")
print("It worked!")
# END-LASER-MULTI-FACE
