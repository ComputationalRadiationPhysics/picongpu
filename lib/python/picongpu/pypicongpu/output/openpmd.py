# This file is part of PIConGPU.
# Copyright 2021-2025 PIConGPU contributors
# Authors: Masoud Afshari
# License: GPLv3+

from .timestepspec import TimeStepSpec
from .plugin import Plugin
from .. import util
from .openpmd_source import (
    ChargeDensity,
    BoundElectronDensity,
    Counter,
    Density,
    Energy,
    EnergyDensity,
    EnergyDensityCutoff,
    LarmorPower,
    MacroCounter,
    MidCurrentDensityComponent,
    Momentum,
    MomentumDensity,
    WeightedVelocity,
)
import typeguard
import typing


@typeguard.typechecked
class Source:
    sources = util.build_typesafe_property(
        typing.List[
            typing.Union[
                str,
                ChargeDensity,
                BoundElectronDensity,
                Counter,
                Density,
                Energy,
                EnergyDensity,
                EnergyDensityCutoff,
                LarmorPower,
                MacroCounter,
                MidCurrentDensityComponent,
                Momentum,
                MomentumDensity,
                WeightedVelocity,
            ]
        ]
    )

    def __init__(
        self,
        sources: typing.List[
            typing.Union[
                str,
                ChargeDensity,
                BoundElectronDensity,
                Counter,
                Density,
                Energy,
                EnergyDensity,
                EnergyDensityCutoff,
                LarmorPower,
                MacroCounter,
                MidCurrentDensityComponent,
                Momentum,
                MomentumDensity,
                WeightedVelocity,
            ]
        ],
    ):
        self.sources = sources

    def _get_serialized(self) -> typing.List[typing.Any]:
        return [s._get_serialized() if not isinstance(s, str) else s for s in self.sources]


@typeguard.typechecked
class OpenPMD(Plugin):
    period = util.build_typesafe_property(TimeStepSpec)
    source = util.build_typesafe_property(typing.Optional[Source])
    range = util.build_typesafe_property(typing.Optional[str])
    file = util.build_typesafe_property(typing.Optional[str])
    ext = util.build_typesafe_property(typing.Optional[str])
    infix = util.build_typesafe_property(typing.Optional[str])
    json = util.build_typesafe_property(typing.Optional[typing.Union[str, typing.Dict]])
    json_restart = util.build_typesafe_property(typing.Optional[typing.Union[str, typing.Dict]])
    data_preparation_strategy = util.build_typesafe_property(typing.Optional[str])
    toml = util.build_typesafe_property(typing.Optional[str])
    particle_io_chunk_size = util.build_typesafe_property(typing.Optional[int])
    file_access = util.build_typesafe_property(typing.Optional[str])

    _name = "openPMD"

    def __init__(
        self,
        period: TimeStepSpec,
        source: typing.Optional[Source] = None,
        range: typing.Optional[str] = None,
        file: typing.Optional[str] = None,
        ext: typing.Optional[str] = None,
        infix: typing.Optional[str] = None,
        json: typing.Optional[typing.Union[str, typing.Dict]] = None,
        json_restart: typing.Optional[typing.Union[str, typing.Dict]] = None,
        data_preparation_strategy: typing.Optional[str] = None,
        toml: typing.Optional[str] = None,
        particle_io_chunk_size: typing.Optional[int] = None,
        file_access: typing.Optional[str] = None,
    ):
        self.period = period
        self.source = source or Source(["species_all", "fields_all"])
        self.range = range or ":,:,:"
        self.file = file
        self.ext = ext or "bp"
        self.infix = infix or "_%06T"
        self.json = json or {}
        self.json_restart = json_restart or {}
        self.data_preparation_strategy = data_preparation_strategy or "doubleBuffer"
        self.toml = toml
        self.particle_io_chunk_size = particle_io_chunk_size
        self.file_access = file_access or "create"

    def _get_serialized(self) -> typing.Dict:
        result = {
            "period": self.period._get_serialized(),
            "source": self.source._get_serialized(),
            "range": self.range,
            "file": self.file,
            "ext": self.ext,
            "infix": self.infix,
            "json": self.json,
            "json_restart": self.json_restart,
            "data_preparation_strategy": self.data_preparation_strategy,
            "toml": self.toml,
            "particle_io_chunk_size": self.particle_io_chunk_size,
            "file_access": self.file_access,
        }
        return {k: v for k, v in result.items() if v is not None}
