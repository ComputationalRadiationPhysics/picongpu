"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field

from picongpu.pypicongpu import memory as pypicongpu_memory

from ._copy_attributes import converts_to


def _non_negative(values):
    if any(x < 0 for x in values):
        raise ValueError(f"All values must be non-negative (>= 0). You gave {values}.")
    return values


@converts_to(pypicongpu_memory.MemoryConfig)
class MemoryConfig(BaseModel):
    """
    Memory / exchange-buffer knobs exposed at the PICMI level and rendered into
    ``include/picongpu/param/memory.param``.

    This is the user-facing grouping of the memory knobs. All values are raw
    byte counts; use the byte-size constants from :mod:`picongpu.picmi.constants`
    for readability, e.g. ``reserved_gpu_memory_size=350 * MiB``. The
    human-readable rendering (e.g. ``350 * 1024 * 1024``) is a pypicongpu concern
    and is applied by :class:`picongpu.pypicongpu.memory.MemoryConfig` on
    conversion (see ``get_as_pypicongpu``). ``super_cell_size`` intentionally
    lives on the grid, not here.
    """

    reserved_gpu_memory_size: Annotated[int, Field(ge=0)] = 350 * 1024 * 1024
    """reserved GPU-internal memory, in bytes (default 350 MiB)."""

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

    ref_local_dom_size: Annotated[tuple[int, int, int], BeforeValidator(_non_negative)] = (0, 0, 0)
    """reference local domain size for exchange scaling; three non-negative ints (0 = no scaling)."""

    dir_scaling_factor: Annotated[tuple[float, float, float], BeforeValidator(_non_negative)] = (0.0, 0.0, 0.0)
    """per-direction scaling rate for the exchange buffers; three non-negative floats (0.0 = no scaling)."""

    field_tmp_support_gather_communication: bool = True
    """whether ``FieldTmp`` may gather neighbor ("ghost"/"halo") information across devices."""
