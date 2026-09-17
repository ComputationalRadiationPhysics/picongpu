"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Annotated

from pydantic import BaseModel, Field, BeforeValidator, PlainSerializer, field_serializer

from picongpu.pypicongpu.grid import serialise_vec


def _non_negative(values):
    if any(x < 0 for x in values):
        raise ValueError(f"All values must be non-negative (>= 0). You gave {values}.")
    return values


def _human_bytes(value: int) -> str:
    if value % (1024 * 1024) == 0:
        return f"{value // (1024 * 1024)} * 1024 * 1024"
    if value % 1024 == 0:
        return f"{value // 1024} * 1024"
    return str(value)


class MemoryConfig(BaseModel):
    """
    Memory / exchange-buffer knobs rendered into ``include/picongpu/param/memory.param``.

    ``reserved_gpu_memory_size`` is given in MiB (rendered as ``<mib> * 1024 * 1024`` bytes);
    the ``bytes_*`` exchange sizes are raw byte counts (rendered human-readable, e.g.
    ``1 * 1024 * 1024`` / ``32 * 1024``); ``ref_local_dom_size`` are three
    non-negative ints (0 = no scaling); ``dir_scaling_factor`` are three floats (0.0 = no
    scaling). ``super_cell_size`` intentionally lives on the grid, not here.
    """

    reserved_gpu_memory_size: Annotated[int, Field(ge=0)] = 350
    """reserved GPU-internal memory, in MiB (rendered as ``<mib> * 1024 * 1024`` bytes)."""

    bytes_exchange_x: Annotated[int, Field(gt=0)] = 1 * 1024 * 1024
    """exchange buffer bytes for the x direction (default 1 MiB)."""

    bytes_exchange_y: Annotated[int, Field(gt=0)] = 3 * 1024 * 1024
    """exchange buffer bytes for the y direction (default 3 MiB)."""

    bytes_exchange_z: Annotated[int, Field(gt=0)] = 1 * 1024 * 1024
    """exchange buffer bytes for the z direction (default 1 MiB)."""

    bytes_edges: Annotated[int, Field(gt=0)] = 32 * 1024
    """exchange buffer bytes for edges (default 32 KiB)."""

    bytes_corner: Annotated[int, Field(gt=0)] = 8 * 1024
    """exchange buffer bytes for corners (default 8 KiB)."""

    ref_local_dom_size: Annotated[
        tuple[int, int, int], BeforeValidator(_non_negative), PlainSerializer(serialise_vec, return_type=dict)
    ] = (0, 0, 0)
    """reference local domain size for exchange scaling; three non-negative ints (0 = no scaling)."""

    dir_scaling_factor: Annotated[
        tuple[float, float, float], BeforeValidator(_non_negative), PlainSerializer(serialise_vec, return_type=dict)
    ] = (0.0, 0.0, 0.0)
    """per-direction scaling rate for the exchange buffers; three non-negative floats (0.0 = no scaling)."""

    field_tmp_support_gather_communication: bool = True
    """whether ``FieldTmp`` may gather neighbor ("ghost"/"halo") information across devices."""

    @field_serializer("reserved_gpu_memory_size", return_type=str)
    def _render_reserved(self, value: int) -> str:
        return f"{value} * 1024 * 1024"

    @field_serializer(
        "bytes_exchange_x", "bytes_exchange_y", "bytes_exchange_z", "bytes_edges", "bytes_corner", return_type=str
    )
    def _render_bytes(self, value: int) -> str:
        return _human_bytes(value)
