"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import typeguard


@typeguard.typechecked
class Solver:
    """
    represents a field solver

    Parent class for type safety, does not contain anything.
    """

    pass
