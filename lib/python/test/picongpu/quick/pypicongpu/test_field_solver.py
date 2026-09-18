"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from unittest import TestCase

import pytest
from pydantic import ValidationError

from picongpu.pypicongpu import field_solver


class TestFieldSolver(TestCase):
    """the Maxwell solver models render their C++ template alias via `name`"""

    def test_fixed_order_solver_names(self):
        # Yee/CKC/None carry no parameters and render as plain aliases
        assert field_solver.YeeSolver().name == "Yee"
        assert field_solver.LeheSolver().name == "Lehe<>"
        assert field_solver.CKCSolver().name == "CKC"
        assert field_solver.NoneSolver().name == "None"

    def test_arbitrary_order_fdt_name(self):
        # the C++ template argument is the number of neighbors (order // 2)
        for neighbors in (1, 2, 3, 4, 5):
            solver = field_solver.ArbitraryOrderFDTDSolver(neighbors=neighbors)
            assert solver.name == f"ArbitraryOrderFDTD<{neighbors}>"

    def test_arbitrary_order_fdt_requires_neighbors(self):
        # neighbors is required and at least 1
        with pytest.raises(ValidationError):
            field_solver.ArbitraryOrderFDTDSolver()
        with pytest.raises(ValidationError):
            field_solver.ArbitraryOrderFDTDSolver(neighbors=0)

    def test_any_solver_union(self):
        # the AnySolver union now covers all five Maxwell solvers
        assert len(field_solver.AnySolver.__args__) == 5
