"""
This file is part of the PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

import pydantic
import typing

from .rendering import RenderedObject


class MovingWindow(RenderedObject, pydantic.BaseModel):
    move_point: float
    """
    point a light ray reaches in y from the left border until we begin sliding the simulation window with the speed of
    light

    in multiples of the simulation window size

    @attention if moving window is active, one gpu in y direction is reserved for initializing new spaces,
        thereby reducing the simulation window size according
    """

    stop_iteration: typing.Optional[int]
    """iteration, at which to stop moving the simulation window"""

    def check(self) -> None:
        if self.move_point < 0.0:
            raise ValueError("window_move point must be >= 0.")
        if self.stop_iteration <= 0:
            raise ValueError("stop iteration must be > 0.")

    def _get_serialized(self) -> dict:
        return {"move_point": self.move_point, "stop_iteration": self.stop_iteration}
