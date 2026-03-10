"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

from ....rendering.pmaccprinter import PMAccPrinter


class FreeFormula(BaseModel):
    type_freeformula: Literal[True] = True
    function_body: Annotated[str, BeforeValidator(PMAccPrinter().doprint)] = Field(alias="density_expression")
