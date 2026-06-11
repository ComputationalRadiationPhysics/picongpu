# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""


"""

from datetime import timedelta
from typing import Annotated

from pydantic import BaseModel, PlainSerializer, field_validator

from .rendering import RenderedObject


def serialise_timedelta(value):
    HOUR = timedelta(hours=1.0)
    MINUTE = timedelta(minutes=1.0)
    SECOND = timedelta(seconds=1.0)

    hours, rest = divmod(value, HOUR)
    minutes, rest = divmod(rest, MINUTE)
    seconds, _ = divmod(rest, SECOND)
    return "{:d}:{:02d}:{:02d}".format(hours, minutes, seconds)


class Walltime(RenderedObject, BaseModel):
    walltime: Annotated[timedelta, PlainSerializer(serialise_timedelta)]
    """time after which the cluster scheduler will stop the simulation"""

    @field_validator("walltime", mode="after")
    @classmethod
    def check(cls, value) -> None:
        if value.total_seconds() <= 0.0:
            raise ValueError("walltime must be > 0.")
        return value
