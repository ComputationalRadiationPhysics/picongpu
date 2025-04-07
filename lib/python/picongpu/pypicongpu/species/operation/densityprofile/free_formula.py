"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .densityprofile import DensityProfile
import typeguard


from sympy.printing.cxx import cxx_code_printers


class AlpakaPrinter(cxx_code_printers["c++17"]):
    pass


@typeguard.typechecked
class FreeFormula(DensityProfile):
    _name = "freeformula"

    def __init__(self, density_expression) -> None:
        self.density_expression = density_expression

    def check(self):
        pass

    def _get_serialized(self) -> dict | None:
        return {"function_body": AlpakaPrinter().doprint(self.density_expression)}
