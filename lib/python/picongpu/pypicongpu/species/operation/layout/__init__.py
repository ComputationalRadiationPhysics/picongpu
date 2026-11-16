"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from .one_position import OnePosition as OnePosition
from .quiet import Quiet as Quiet
from .random import Random as Random

AnyLayout = Random | Quiet | OnePosition
