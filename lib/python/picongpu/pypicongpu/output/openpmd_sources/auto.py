"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .... import util
from .source_base import SourceBase
import typeguard
import typing


@typeguard.typechecked
class Auto(SourceBase):
    filter = util.build_typesafe_property(typing.Optional[str])

    def __init__(self, filter: typing.Optional[str] = None):
        self.filter = filter
        self.check()

    def check(self) -> None:
        # Validate filter parameter
        if self.filter is not None and not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string or None, got {type(self.filter)}")

    def _get_serialized(self) -> typing.Dict:
        # Return serialized representation
        self.check()
        return {"filter": self.filter}
