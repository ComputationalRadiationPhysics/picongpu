"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from .. import util
from .timestepspec import TimeStepSpec
from .rangespec import RangeSpec
from .plugin import Plugin
from .openpmd_sources.source_base import SourceBase

import typeguard
import typing
from typing import Optional, List, Literal, Dict, Union


@typeguard.typechecked
class OpenPMD(Plugin):
    period = util.build_typesafe_property(TimeStepSpec)
    source = util.build_typesafe_property(Optional[List[SourceBase]])
    range = util.build_typesafe_property(Optional[RangeSpec])
    file = util.build_typesafe_property(Optional[str])
    ext = util.build_typesafe_property(Optional[Literal["bp", "h5", "sst"]])
    infix = util.build_typesafe_property(Optional[str])
    json = util.build_typesafe_property(Union[str, Dict, None])
    json_restart = util.build_typesafe_property(Union[str, Dict, None])
    data_preparation_strategy = util.build_typesafe_property(
        Optional[Literal["doubleBuffer", "adios", "mappedMemory", "hdf5"]]
    )
    toml = util.build_typesafe_property(Optional[str])
    particle_io_chunk_size = util.build_typesafe_property(Optional[int])
    file_writing = util.build_typesafe_property(Optional[Literal["create", "append"]])

    _name = "openpmd"

    def __init__(
        self,
        period: TimeStepSpec,
        source: Optional[List[SourceBase]] = None,
        range: Optional[RangeSpec] = None,
        file: Optional[str] = None,
        ext: Optional[Literal["bp", "h5", "sst"]] = "bp",
        infix: Optional[str] = "NULL",
        json: Optional[Union[str, Dict]] = None,
        json_restart: Optional[Union[str, Dict]] = None,
        data_preparation_strategy: Optional[Literal["doubleBuffer", "adios", "mappedMemory", "hdf5"]] = None,
        toml: Optional[str] = None,
        particle_io_chunk_size: Optional[int] = None,
        file_writing: Optional[Literal["create", "append"]] = "create",
    ):
        self.period = period
        self.source = source
        self.range = range
        self.file = file
        self.ext = ext
        self.infix = infix
        self.json = None if json is None or json == {} else json
        self.json_restart = None if json_restart is None or json_restart == {} else json_restart
        self.data_preparation_strategy = data_preparation_strategy
        self.toml = toml
        self.particle_io_chunk_size = particle_io_chunk_size
        self.file_writing = file_writing
        self.check()

    def check(self) -> None:
        if self.file is not None and len(self.file.strip()) == 0:
            raise ValueError("file must be a non-empty string")
        if self.particle_io_chunk_size is not None and self.particle_io_chunk_size < 1:
            raise ValueError("particle_io_chunk_size (in MiB) must be positive")
        if self.ext == "sst" and self.infix is not None and self.infix != "NULL":
            raise ValueError("infix must be 'NULL' when ext is 'sst'")
        if self.source is not None and not all(isinstance(s, SourceBase) for s in self.source):
            raise ValueError("source must be a list of SourceBase objects")

    def _get_serialized(self) -> typing.Dict:
        self.check()

        return {
            "period": self.period.get_rendering_context(),
            "source": [s._get_serialized() for s in self.source] if self.source is not None else None,
            "range": self.range._get_serialized() if self.range else None,
            "file": self.file,
            "ext": self.ext,
            "infix": self.infix,
            "json": self.json,
            "json_restart": self.json_restart,
            "data_preparation_strategy": self.data_preparation_strategy,
            "toml": self.toml,
            "particle_io_chunk_size": self.particle_io_chunk_size,
            "file_writing": self.file_writing,
        }
