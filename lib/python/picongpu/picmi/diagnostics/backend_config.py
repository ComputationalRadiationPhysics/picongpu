"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from enum import Enum

from picongpu.pypicongpu.output.openpmd_plugin import OpenPMDConfig, RangeSpecEntry
from picongpu.pypicongpu.output.openpmd_plugin import RangeSpec as PyPIConGPURangeSpec

# If there should ever be another option,
# this type alias could be expanded via type union.
BackendConfig = OpenPMDConfig


class RangeSpecUnit(Enum):
    CELLS = "Cells"


def _apply_units(iterable, unit):
    if unit != RangeSpecUnit.CELLS:
        raise ValueError(f"Unknown RangeSpecUnit. You gave: {unit=}.")
    return tuple(iterable)


class RangeSpec(PyPIConGPURangeSpec):
    def __init__(self, *args, **kwargs):
        unit = kwargs.pop("unit", RangeSpecUnit.CELLS)
        if len(args) == 3:
            data = args
        elif len(args) == 0:
            data = (kwargs.pop("x", None), kwargs.pop("y", None), kwargs.pop("z", None))
        else:
            raise ValueError(f"Unknown RangeSpec construction. You gave {args=}.")
        data = _apply_units(map(lambda x: RangeSpecEntry(data=x), data), unit)
        return super().__init__(data=data, **kwargs)
