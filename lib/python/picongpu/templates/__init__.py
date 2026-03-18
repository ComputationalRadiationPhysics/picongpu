"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import lru_cache
from pathlib import Path


@lru_cache
def path():
    return Path(__file__).parent.absolute()
