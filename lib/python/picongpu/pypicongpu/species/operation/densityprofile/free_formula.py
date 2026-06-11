"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field

from ....rendering.pmaccprinter import PMAccPrinter


class FreeFormula(BaseModel):
    type_freeformula: Literal[True] = True
    function_body: Annotated[str, BeforeValidator(PMAccPrinter().doprint)] = Field(alias="density_expression")
