"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from functools import lru_cache
from typing import Literal
from pathlib import Path


@lru_cache
def path(component: Literal["bin", "etc", "include"] | None = None):
    if component is None:
        return path("bin").parent
    here = Path(__file__).parent.absolute()
    alternative_location_in_source = here.parents[3]
    expected_components = "pic-create" if component == "bin" else "picongpu"
    try:
        try:
            # If we have been installed,
            # the content has been copied into this directory.
            return next(here.glob(component))
        except StopIteration:
            # We have not been installed (properly),
            # but maybe we're in a source directory,
            # so we know where to look for stuff:
            return next(alternative_location_in_source.glob(f"{component}/{expected_components}")).parent
    except StopIteration:
        message = (
            f"Our heuristic for finding PIConGPU core {component=} has failed. "
            f"We have looked for {here / component} and {alternative_location_in_source / component / expected_components} without success."
        )
        raise FileNotFoundError(message)
