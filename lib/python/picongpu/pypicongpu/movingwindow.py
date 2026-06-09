# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of the PIConGPU.
Copyright 2024-2025 PIConGPU contributors
Authors: Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from typing import Annotated

from pydantic import BaseModel, Field

from .rendering import RenderedObject


class MovingWindow(RenderedObject, BaseModel):
    move_point: Annotated[float, Field(..., ge=0.0)]
    """
    point a light ray reaches in y from the left border until we begin sliding the simulation window with the speed of
    light

    in multiples of the simulation window size

    @attention if moving window is active, one gpu in y direction is reserved for initializing new spaces,
        thereby reducing the simulation window size according
    """

    stop_iteration: Annotated[int, Field(..., gt=0.0)] | None
    """iteration, at which to stop moving the simulation window"""
