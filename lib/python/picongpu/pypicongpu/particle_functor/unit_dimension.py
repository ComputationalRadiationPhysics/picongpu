"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from pydantic import BaseModel, PrivateAttr, model_serializer, model_validator


class UnitDimension(BaseModel):
    _num_unit_dimensions: int = PrivateAttr(7)
    unit_dimension: list = _num_unit_dimensions.default * [0.0]

    @model_validator(mode="after")
    def check(self):
        if len(self.unit_dimension) != self._num_unit_dimensions:
            raise ValueError(
                f"Unit dimension vector has {len(self.unit_dimension)=} but {self._num_unit_dimensions=}. They must match."
            )
        return self

    @model_serializer(mode="plain")
    def translate_to_cpp(self) -> str:
        return f"std::array<double, {self._num_unit_dimensions}u>{{{','.join(map(str, self.unit_dimension))}}}"
