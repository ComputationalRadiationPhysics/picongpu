#!/usr/bin/env python
# /// script
# dependencies = [
#   "picongpu @ git+https://github.com/ComputationalRadiationPhysics/picongpu@dev#subdirectory=lib/python"
# ]
# ///
"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+

Generates a simulation setup and inspects the RO-Crate specification version
recorded in its metadata. The version is the ``rocrate.info.version`` rc
parameter, exposed as ``rc_params.rocrate_info.version`` and defaulting to
``1.3``.
"""

# BEGIN-ROCRATE-VERSION
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from picongpu import picmi, rc_params

grid = picmi.Cartesian3DGrid(
    number_of_cells=[32, 32, 32],
    lower_bound=[0, 0, 0],
    upper_bound=[1e-6, 1e-6, 1e-6],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", grid=grid)
sim = picmi.Simulation(time_step_size=17, max_steps=4, solver=solver)

# the RO-Crate specification version is an rc parameter (default "1.3");
# rc_params.rocrate_info is its resolved view with the defaults applied
print("configured RO-Crate version:", rc_params.rocrate_info.version)

with TemporaryDirectory() as setup_dir:
    sim.write_input_file(setup_dir, exist_ok=True)
    metadata = json.loads((Path(setup_dir) / "ro-crate-metadata.json").read_text())
    assert metadata["@context"] == "https://w3id.org/ro/crate/1.3/context"
    print("written RO-Crate context:", metadata["@context"])

# select another supported version before generating: the rc parameter lives
# in the ``rocrate.info`` section, which backs the resolved ``rocrate_info``
rc_params["rocrate"] = {"info": {"version": "1.2"}}
assert rc_params["rocrate"]["info"]["version"] == rc_params.rocrate_info.version == "1.2"
print("overridden RO-Crate version:", rc_params.rocrate_info.version)

with TemporaryDirectory() as setup_dir:
    sim.write_input_file(setup_dir, exist_ok=True)
    metadata = json.loads((Path(setup_dir) / "ro-crate-metadata.json").read_text())
    assert metadata["@context"] == "https://w3id.org/ro/crate/1.2/context"
    print("overridden RO-Crate context:", metadata["@context"])
# END-ROCRATE-VERSION
