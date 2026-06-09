# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

from .Yee import YeeSolver as YeeSolver
from .Lehe import LeheSolver as LeheSolver

AnySolver = YeeSolver | LeheSolver
