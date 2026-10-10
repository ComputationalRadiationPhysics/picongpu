"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from picmistandard import PICMI_FromFileDistribution
from pydantic import Field


class FromFileDistribution(PICMI_FromFileDistribution):
    """Load the particles of a species from an external openPMD file.

    This is the PICMI-standard :class:`picmistandard.PICMI_FromFileDistribution`
    (which carries only ``file_path``) extended with the openPMD ``iteration``
    to read (default ``0``).

    The file must be a standard-compliant openPMD particle file providing
    ``position`` and ``positionOffset`` (the latter either constant or
    non-constant, in both cases its ``unitSI`` conversion is respected),
    ``momentum`` and ``weighting``. ``particlePatches`` are used as a fast path
    when present in PIConGPU's cell-index layout; otherwise the particles are
    placed by their global position, which is slower and less memory-efficient.

    Unlike a density distribution, the positions of the particles are given by
    the file, so this distribution must NOT be combined with a
    :class:`~picongpu.picmi.layout.Layout`; doing so raises a ``ValueError``.
    """

    iteration: int = Field(default=0, ge=0)
    """openPMD iteration to read; defaults to 0"""
