"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
License: GPLv3+
"""

import os
import re
import tempfile
from unittest import TestCase

from pydantic import TypeAdapter

from picongpu import picmi
from picongpu.picmi.diagnostics import AnyDiagnostic, FieldEnergyMonitor, TimeStepSpec
from picongpu.pypicongpu.output import AnyPlugin, FieldEnergyMonitor as PyPIConGPUFieldEnergyMonitor


class TestFieldEnergyMonitor(TestCase):
    def test_converts_to_pypicongpu(self):
        monitor = FieldEnergyMonitor(period=TimeStepSpec[:16:2])

        converted = monitor.get_as_pypicongpu(time_step_size=1, num_steps=17)

        assert isinstance(converted, PyPIConGPUFieldEnergyMonitor)
        assert converted.type_fieldenergymonitor is True
        assert converted.period.model_dump() == {"specs": [{"start": 0, "stop": 16, "step": 2}]}

    def test_period_is_required(self):
        with self.assertRaises(Exception):
            FieldEnergyMonitor()

    def test_has_no_species(self):
        # FieldEnergyMonitor is a free diagnostic, not per-species
        assert "species" not in FieldEnergyMonitor.model_fields
        assert not hasattr(FieldEnergyMonitor(period=TimeStepSpec[0]), "species")

    def test_registered_in_output_union(self):
        # Simulation.output is validated against AnyPlugin; the diagnostic must be accepted there
        monitor = FieldEnergyMonitor(period=TimeStepSpec[1:10:3])
        converted = monitor.get_as_pypicongpu(time_step_size=1, num_steps=17)

        validated = TypeAdapter(AnyPlugin).validate_python(converted.model_dump())

        assert isinstance(validated, PyPIConGPUFieldEnergyMonitor)
        assert isinstance(monitor, AnyDiagnostic) or "FieldEnergyMonitor" in str(AnyDiagnostic)

    def test_rendered_n_cfg_emits_fields_energy(self):
        grid = picmi.Cartesian3DGrid(
            number_of_cells=[32, 32, 32],
            lower_bound=[0, 0, 0],
            upper_bound=[3.2e-6, 3.2e-6, 3.2e-6],
            lower_boundary_conditions=["open", "open", "open"],
            upper_boundary_conditions=["open", "open", "open"],
        )
        solver = picmi.ElectromagneticSolver(method="Yee", grid=grid, cfl=1.0)
        sim = picmi.Simulation(time_step_size=None, max_steps=17, solver=solver)
        sim.diagnostics = [FieldEnergyMonitor(period=TimeStepSpec[1:10:3])]

        with tempfile.TemporaryDirectory() as outdir:
            sim.write_input_file(outdir, exist_ok=True)

            candidates = [
                os.path.join(root, name) for root, _, names in os.walk(outdir) for name in names if name == "N.cfg"
            ]
            assert candidates, f"no rendered N.cfg found under {outdir}"

            with open(candidates[0]) as cfg:
                content = cfg.read()

        # the option must use the hardcoded prefix and the requested period, with no species
        match = re.search(r"--fields_energy\.period (\S+)", content)
        assert match is not None, "fields_energy options not found in rendered N.cfg"
        assert match.group(1) == "1:10:3"
        assert "--fields_energy.species" not in content
