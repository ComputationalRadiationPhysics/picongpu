"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

import numpy as np

NUMBER_OF_CELLS = [64, 64, 32]
UPPER_BOUNDARY = np.array([64.0, 66.0, 74.0])
CELL_SIZE = UPPER_BOUNDARY / NUMBER_OF_CELLS

ALL_ORIGINS = ["total", "global", "local", "moving_window", "local_with_guards"]
ALL_ORIGINS_WITHOUT_GUARDS = [origin for origin in ALL_ORIGINS if not origin.endswith("guards")]
ALL_PRECISIONS = ["cell", "sub_cell"]
ALL_UNITS = ["cell", "si", "pic"]
NUMBER_OF_GUARD_CELLS = [8, 8, 4]
EPSILON = 1.0e-5
