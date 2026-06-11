"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
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
