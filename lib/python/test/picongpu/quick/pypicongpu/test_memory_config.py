"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase

import pytest
from pydantic import ValidationError

from picongpu import picmi, templates
from picongpu.pypicongpu.memory import MemoryConfig
from picongpu.pypicongpu.rendering import Renderer


def _grid():
    return picmi.Cartesian3DGrid(
        number_of_cells=[32, 32, 32],
        lower_bound=[0, 0, 0],
        upper_bound=[1, 1, 1],
        lower_boundary_conditions=["open", "open", "periodic"],
        upper_boundary_conditions=["open", "open", "periodic"],
    )


def _sim(**sim_kwargs):
    solver = picmi.ElectromagneticSolver(method="Yee", grid=_grid())
    return picmi.Simulation(time_step_size=17, max_steps=4, solver=solver, **sim_kwargs)


def _rendered(template_name, sim):
    context = Renderer.get_context_preprocessed(sim.get_as_pypicongpu().get_rendering_context())
    path = templates.path() / "include/picongpu/param" / template_name
    with open(path) as file:
        return Renderer.get_rendered_template(context, file.read())


class TestMemoryConfigDefaults(TestCase):
    def test_defaults_round_trip(self):
        config = MemoryConfig()
        assert config.reserved_gpu_memory_size == 350
        assert (config.bytes_exchange_x, config.bytes_exchange_y, config.bytes_exchange_z) == (
            1 * 1024 * 1024,
            3 * 1024 * 1024,
            1 * 1024 * 1024,
        )
        assert (config.bytes_edges, config.bytes_corner) == (32 * 1024, 8 * 1024)
        assert config.ref_local_dom_size == (0, 0, 0)
        assert config.dir_scaling_factor == (0.0, 0.0, 0.0)
        assert config.field_tmp_support_gather_communication is True

    def test_reserved_renders_mib(self):
        dumped = MemoryConfig().model_dump(mode="json")
        assert dumped["reserved_gpu_memory_size"] == "350 * 1024 * 1024"

    def test_reserved_mib_custom(self):
        dumped = MemoryConfig(reserved_gpu_memory_size=100).model_dump(mode="json")
        assert dumped["reserved_gpu_memory_size"] == "100 * 1024 * 1024"

    def test_vec_serialised_to_xyz_dict(self):
        dumped = MemoryConfig(ref_local_dom_size=(2, 3, 4), dir_scaling_factor=(0.5, 0.25, 1.0)).model_dump(mode="json")
        assert dumped["ref_local_dom_size"] == {"x": 2, "y": 3, "z": 4}
        assert dumped["dir_scaling_factor"] == {"x": 0.5, "y": 0.25, "z": 1.0}

    def test_super_cell_size_not_in_memory(self):
        # super_cell_size deliberately stays on the grid, not in MemoryConfig
        assert "super_cell_size" not in MemoryConfig.model_fields


class TestMemoryConfigValidation(TestCase):
    def test_bytes_must_be_positive(self):
        for field in ("bytes_exchange_x", "bytes_exchange_y", "bytes_exchange_z", "bytes_edges", "bytes_corner"):
            with pytest.raises(ValidationError):
                MemoryConfig(**{field: 0})
            with pytest.raises(ValidationError):
                MemoryConfig(**{field: -1})

    def test_reserved_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            MemoryConfig(reserved_gpu_memory_size=-1)

    def test_ref_local_dom_size_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            MemoryConfig(ref_local_dom_size=(0, -1, 0))
        assert MemoryConfig(ref_local_dom_size=(0, 0, 0)).ref_local_dom_size == (0, 0, 0)

    def test_dir_scaling_factor_must_be_non_negative(self):
        with pytest.raises(ValidationError):
            MemoryConfig(dir_scaling_factor=(0.0, -0.1, 0.0))


class TestPrecisionOverrides(TestCase):
    def test_default_is_core(self):
        sim = _sim()
        assert (sim.picongpu_precision_sqrt, sim.picongpu_precision_exp, sim.picongpu_precision_trig) == (
            "core",
            "core",
            "core",
        )
        p = sim.get_as_pypicongpu()
        assert (p.precisionSqrt, p.precisionExp, p.precisionTrigonometric) == (
            "precisionPIConGPU",
            "precisionPIConGPU",
            "precisionPIConGPU",
        )

    def test_values_map_to_namespaces(self):
        sim = _sim(
            picongpu_precision=64,
            picongpu_precision_sqrt=64,
            picongpu_precision_exp="core",
            picongpu_precision_trig=32,
        )
        p = sim.get_as_pypicongpu()
        assert (p.precisionSqrt, p.precisionExp, p.precisionTrigonometric) == (
            "precision64Bit",
            "precisionPIConGPU",
            "precision32Bit",
        )

    def test_invalid_precision_rejected(self):
        for value in (128, 0, 16):
            with pytest.raises(ValidationError):
                _sim(picongpu_precision_sqrt=value)


class TestMemoryTranslation(TestCase):
    def test_default_is_memory_config(self):
        assert isinstance(_sim().picongpu_memory, MemoryConfig)

    def test_custom_memory_flows_through(self):
        sim = _sim(
            picongpu_memory=MemoryConfig(
                reserved_gpu_memory_size=100,
                ref_local_dom_size=(2, 3, 4),
                dir_scaling_factor=(0.5, 0.25, 1.0),
                field_tmp_support_gather_communication=False,
            )
        )
        mem = sim.get_as_pypicongpu().memory_config
        assert mem.reserved_gpu_memory_size == 100
        assert mem.ref_local_dom_size == (2, 3, 4)
        assert mem.dir_scaling_factor == (0.5, 0.25, 1.0)
        assert mem.field_tmp_support_gather_communication is False

    def test_invalid_memory_rejected(self):
        with pytest.raises(ValidationError):
            _sim(picongpu_memory=MemoryConfig(bytes_edges=0))


class TestTemplateRendering(TestCase):
    def test_memory_param_defaults(self):
        rendered = _rendered("memory.param.mustache", _sim())
        assert "constexpr size_t reservedGpuMemorySize = 350 * 1024 * 1024;" in rendered
        assert "static constexpr uint32_t BYTES_EXCHANGE_X = 1 * 1024 * 1024;" in rendered
        assert "static constexpr uint32_t BYTES_EXCHANGE_Y = 3 * 1024 * 1024;" in rendered
        assert "static constexpr uint32_t BYTES_EXCHANGE_Z = 1 * 1024 * 1024;" in rendered
        assert "static constexpr uint32_t BYTES_EDGES = 32 * 1024;" in rendered
        assert "static constexpr uint32_t BYTES_CORNER = 8 * 1024;" in rendered
        assert "using REF_LOCAL_DOM_SIZE = mCT::Int<0, 0, 0>;" in rendered
        assert "const std::array<float_X, 3> DIR_SCALING_FACTOR = { 0.0, 0.0, 0.0 };" in rendered
        assert "constexpr bool fieldTmpSupportGatherCommunication = true;" in rendered
        # super_cell_size stays in the grid
        assert "mCT::Int<8, 8, 4>" in rendered

    def test_memory_param_custom(self):
        sim = _sim(
            picongpu_memory=MemoryConfig(
                reserved_gpu_memory_size=100,
                bytes_exchange_x=2 * 1024 * 1024,
                ref_local_dom_size=(2, 3, 4),
                dir_scaling_factor=(0.5, 0.25, 1.0),
                field_tmp_support_gather_communication=False,
            )
        )
        rendered = _rendered("memory.param.mustache", sim)
        assert "constexpr size_t reservedGpuMemorySize = 100 * 1024 * 1024;" in rendered
        assert "static constexpr uint32_t BYTES_EXCHANGE_X = 2 * 1024 * 1024;" in rendered
        assert "using REF_LOCAL_DOM_SIZE = mCT::Int<2, 3, 4>;" in rendered
        assert "const std::array<float_X, 3> DIR_SCALING_FACTOR = { 0.5, 0.25, 1.0 };" in rendered
        assert "constexpr bool fieldTmpSupportGatherCommunication = false;" in rendered

    def test_precision_param_defaults(self):
        rendered = _rendered("precision.param.mustache", _sim())
        assert "namespace precisionPIConGPU = precision32Bit;" in rendered
        assert "namespace precisionSqrt = precisionPIConGPU;" in rendered
        assert "namespace precisionExp = precisionPIConGPU;" in rendered
        assert "namespace precisionTrigonometric = precisionPIConGPU;" in rendered

    def test_precision_param_overrides(self):
        sim = _sim(
            picongpu_precision=64,
            picongpu_precision_sqrt=32,
            picongpu_precision_exp=64,
            picongpu_precision_trig="core",
        )
        rendered = _rendered("precision.param.mustache", sim)
        assert "namespace precisionPIConGPU = precision64Bit;" in rendered
        assert "namespace precisionSqrt = precision32Bit;" in rendered
        assert "namespace precisionExp = precision64Bit;" in rendered
        assert "namespace precisionTrigonometric = precisionPIConGPU;" in rendered
