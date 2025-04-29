"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

import typeguard
import typing


@typeguard.typechecked
class Auto:
    filter = property(lambda self: self._filter)

    def __init__(self, filter: typing.Optional[str] = None):
        self._filter = filter
        self.check()

    def check(self) -> None:
        if self._filter is not None and not isinstance(self._filter, str):
            raise ValueError(f"Filter must be a string or None, got {type(self._filter)}")

    def _get_serialized(self) -> typing.Dict:
        return {
            "source": [
                {"dataset": "species_all", "filter": self._filter},
                {"dataset": "fields_all", "filter": self._filter},
            ],
            "range": ":,:,:",
            "file": "simOutput",
            "ext": "bp",
            "infix": "NULL",
            "json": {},
            "json_restart": {},
            "data_preparation_strategy": "doubleBuffer",
            "toml": None,
            "particle_io_chunk_size": None,
            "file_access": "create",
        }
