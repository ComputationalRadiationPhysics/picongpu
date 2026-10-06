"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase

import sympy
from picongpu.pypicongpu.rendering.pmaccprinter import PMAccPrinter


class TestPMAccPrinterReservedWords(TestCase):
    def test_cpp_keywords_are_escaped(self):
        # a symbol named like a keyword must be escaped, not emitted verbatim:
        # PMacc compiles with C++20, so sympy's C++17 list is not enough
        printer = PMAccPrinter()
        keywords = [
            "float",
            "int",
            "class",
            "operator",
            "catch",
            "register",
            "char8_t",
            "concept",
            "consteval",
            "constinit",
            "co_await",
            "co_return",
            "co_yield",
            "import",
            "module",
            "requires",
        ]
        for keyword in keywords:
            with self.subTest(keyword=keyword):
                assert printer.doprint(sympy.Symbol(keyword)) == f"{keyword}_"

    def test_non_keyword_is_unchanged(self):
        assert PMAccPrinter().doprint(sympy.Symbol("wavelength")) == "wavelength"

    def test_keyword_in_expression_is_escaped(self):
        # the escaped spelling must also be used inside a composite expression
        rendered = PMAccPrinter().doprint(sympy.sympify("requires*x"))
        assert rendered == "requires_*x"
