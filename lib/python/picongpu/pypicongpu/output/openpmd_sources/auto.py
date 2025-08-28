"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ... import util
from .source_base import SourceBase
import typeguard
import typing


@typeguard.typechecked
class Auto(SourceBase):
    filter = util.build_typesafe_property(str)

    def __init__(self, filter: str = "species_all"):
        self.filter = filter
        self.check()

    def check(self) -> None:
        valid_filters = ["species_all", "fields_all", "custom_filter"]
        if not isinstance(self.filter, str):
            raise ValueError(f"Filter must be a string, got {type(self.filter)}")
        if self.filter not in valid_filters:
            raise ValueError(f"Filter must be one of {valid_filters}, got {self.filter}")

    def _get_serialized(self) -> typing.Dict:
        self.check()
        return {
            "filter": self.filter,
        }
