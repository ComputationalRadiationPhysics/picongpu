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

Configures the openPMD backend through the typed
:class:`~picongpu.picmi.diagnostics.OpenPMDBackendConfig` model:
a global default dataset configuration (blosc compression) with a
pattern-matched exception (no compression for the offset/patch datasets),
plus an HDF5 dataset default.
The resulting nested table is rendered into the plugin's TOML config
under the ``backend_config`` key.
"""

from pathlib import Path

from picongpu import picmi
from picongpu.picmi.diagnostics import (
    NativeFieldDump,
    OpenPMDBackendConfig,
    OpenPMDConfig,
    TS,
)

# BEGIN-OPENPMD-BACKEND-CONFIG
backend_config = OpenPMDBackendConfig(
    backend="adios2",
    iteration_encoding="group_based",
    rank_table="hostname",
    adios2={
        "engine": {"type": "bp5", "parameters": {"BufferGrowthFactor": "1.2"}},
        "dataset": [
            # default entry (no "select"): blosc compression for every dataset
            {"cfg": {"operators": [{"type": "blosc", "parameters": {"clevel": "1"}}]}},
            # pattern-matched exception: skip compression for these datasets
            {"select": [".*positionOffset.*", ".*particlePatches.*"], "cfg": {"operators": []}},
        ],
    },
    hdf5={"dataset": {"chunks": "auto"}},
)
# END-OPENPMD-BACKEND-CONFIG

grid = picmi.Cartesian3DGrid(
    number_of_cells=[32, 32, 32],
    lower_bound=[0, 0, 0],
    upper_bound=[1e-6, 1e-6, 1e-6],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
)
solver = picmi.ElectromagneticSolver(method="Yee", cfl=0.5, grid=grid)
distribution = picmi.UniformDistribution(density=1e23)
layout = picmi.PseudoRandomLayout(n_macroparticles_per_cell=1)
electrons = picmi.Species(name="electrons", particle_type="electron", initial_distribution=distribution)

electric_field_dump = NativeFieldDump(
    fieldname="E",
    period=TS[::5],
    options=OpenPMDConfig(file="simData", backend_config=backend_config),
)

sim = picmi.Simulation(
    max_steps=100,
    solver=solver,
    species=[electrons],
    layouts=[layout],
    diagnostics=[electric_field_dump],
)

sim.run(setup_dir=Path("openpmd_backend_setup"), run_dir=Path("openpmd_backend_run"))

for config in sorted(Path("openpmd_backend_setup").joinpath("etc").glob("openPMD_config_*.toml")):
    print(config.read_text())
