"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from typing import Literal

from pydantic import BaseModel


class PrecisionConfig(BaseModel):
    """
    Per-namespace precision overrides rendered into
    ``include/picongpu/param/precision.param``.

    Each field selects the precision of one special-operation namespace.
    ``"core"`` (default) aliases the core ``precisionPIConGPU`` precision, so a
    64-bit core does not silently downgrade these; ``32``/``64`` force
    ``precision32Bit``/``precision64Bit`` respectively.
    """

    sqrt: Literal[32, 64, "core"] = "core"
    """precision of ``sqrt`` special operations (see ``precision.param``)."""

    exp: Literal[32, 64, "core"] = "core"
    """precision of ``exp`` special operations (see ``precision.param``)."""

    trig: Literal[32, 64, "core"] = "core"
    """precision of trigonometric special operations (see ``precision.param``)."""
