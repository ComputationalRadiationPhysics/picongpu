"""
# SPDX-FileCopyrightText: Julian Lenz
#
# SPDX-License-Identifier: GPL-3.0-or-later
"""

from typing import Annotated
from pydantic import BaseModel, PlainSerializer, field_validator
from ..rendering.renderedobject import RenderedObject


class Spec(BaseModel):
    start: Annotated[int | None, PlainSerializer(lambda x: x if x is not None else 0)]
    stop: Annotated[int | None, PlainSerializer(lambda x: x if x is not None else -1)]
    step: Annotated[int | None, PlainSerializer(lambda x: x if x is not None else 1)]


class TimeStepSpec(RenderedObject, BaseModel):
    specs: list[Spec]

    def __init__(self, *args, **kwargs):
        # allow to give specs as positional argument
        if len(args) > 0 and "specs" not in kwargs:
            kwargs |= {"specs": args[0]}
        super(TimeStepSpec, self).__init__(*args[1:], **kwargs)

    @field_validator("specs", mode="before")
    @classmethod
    def validate_specs(cls, value) -> list[Spec]:
        try:
            return [Spec(start=s.start, stop=s.stop, step=s.step) for s in value]
        except AttributeError:
            return value
