"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Annotated, Literal
from pydantic import BaseModel, BeforeValidator, computed_field

from picongpu.pypicongpu.particle_functor.translate_to_cpp_type import translate_to_cpp_type


class UniformRNGInfo(BaseModel):
    dist: Literal["uniform"] = "uniform"
    return_type: Annotated[str, BeforeValidator(translate_to_cpp_type)]

    @computed_field
    def typename(self) -> str:
        return "pmacc::random::distributions::Uniform"


class NormalRNGInfo(BaseModel):
    dist: Literal["normal"] = "normal"
    return_type: Annotated[str, BeforeValidator(translate_to_cpp_type)]

    @computed_field
    def typename(self) -> str:
        return "pmacc::random::distributions::Normal"


RNGInfo = UniformRNGInfo | NormalRNGInfo
