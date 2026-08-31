#!/bin/bash
#
# This file is part of PIConGPU.
# Copyright 2026 PIConGPU contributors
# Authors: opencode
# License: GPLv3+
#
# Profile check for the "docs-snippets" GitLab job.
#
# Generates a minimal input set with the "bash" preset (no compilation) and
# sources the generated workflow profile, as
# docs/source/python_package/snippets/running_simulation/legacy_workflow.sh
# performs it.
#
# The check lives in a script (instead of an inline heredoc in .gitlab-ci.yml)
# so that it can be executed and tested outside of GitLab.
#
# Must be run with the picongpu python package importable by `python3`
# (e.g. after sourcing share/ci/install/pypicongpu.sh).

set -euo pipefail

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT
cd "$work_dir"
printf 'preset = "bash"\n' > .picongpurc.toml
PIC_RC=$PWD/.picongpurc.toml python3 - <<'PY'
from pathlib import Path

from picongpu import picmi

sim = picmi.Simulation(
    max_steps=1,
    solver=picmi.ElectromagneticSolver(
        method="Yee",
        cfl=0.95,
        grid=picmi.Cartesian3DGrid(
            number_of_cells=[16, 16, 16],
            lower_bound=[0.0, 0.0, 0.0],
            upper_bound=[1e-5, 1e-5, 1e-5],
            lower_boundary_conditions=["periodic", "periodic", "periodic"],
            upper_boundary_conditions=["periodic", "periodic", "periodic"],
        ),
    ),
)
sim.write_input_file(Path("setup"))
PY
source setup/workflow/scripts/picongpu.profile
test -n "$PIC_BACKEND"
test -n "$PICSRC"
echo "profile check OK: PIC_BACKEND=$PIC_BACKEND PICSRC=$PICSRC"
