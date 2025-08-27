"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.openpmd import OpenPMD as PyPIConGPUOpenPMD
from .timestepspec import TimeStepSpec
from .rangespec import RangeSpec
from .openpmd_sources.source_base import SourceBase
from ..species import Species as PICMISpecies

import typeguard
from typing import Optional, Dict, Union, List, Literal, Tuple


@typeguard.typechecked
class OpenPMD:
    """
    openPMD diagnostic output

    This diagnostic writes simulation data (base fields, derived fields and/or particles) to disk using the openPMD
    standard, with configurable periods, data sources and backend settings.

    Parameters
    ----------
    period: TimeStepSpec
        Specification of the time steps for data output, outputs will always be written at the end of a PIC time step.
    source: List[SourceBase], optional
        List of data source objects to include in the dump (e.g., [ChargeDensity(filter="all")]).
        Setting to None will cause an empty dump.
    range: str or RangeSpec, optional
        Contiguous range of cells to dump the base- and derived field for, specified as a RangeSpec object
        Use RangeSpec[start:stop,...] style to specify dimensions (e.g., RangeSpec[0:10, 5:15], RangeSpec[:, :, :])
    file: str, optional
        Relative or absolute file path prefix for openPMD output files. Relative paths are interpreted as relative to the
        simulation output directory, the default value None indicates the PIC code's default.
    ext: str, optional
        File extension controlling the openPMD backend, options are "bp" (default backend ADIOS2), "h5" (HDF5),
        "sst" (ADIOS2/SST for streaming). Default: "bp".
    infix: str, optional
        Filename infix for the iteration layout (e.g., "_%06T"), use "NULL" for the group-based layout,
        ext="sst" requires infix="NULL". Default: "NULL".
    json: str or dict, optional
        openPMD backend configuration as a JSON string, dictionary, or filename (filename must be prepended with "@").
    json_restart: str or dict, optional
        Backend-specific parameters for restarting, as a JSON string, dictionary, or filename (filenames must be
        prepended with "@").
    data_preparation_strategy: str, optional
        Strategy for particle data preparation, options: "doubleBuffer" or "adios" (ADIOS2-based),
        "mappedMemory" or "hdf5" (HDF5-based), the default value None indicates the PIC code default.
    toml: str, optional
        Path to a TOML file for openPMD configuration. Replaces the JSON or keyword configuration.
    particle_io_chunk_size: int, optional
        Size of particle data chunks used in writing (in MiB), reduces host memory footprint for certain backends,
        default "None" indicates the PIC code default.
    file_writing: str, optional
        File writing mode for writing, options: "create" (new files), "append" (for checkpoint-restart workflows).
        Default: "create".
    """

    def __init__(
        self,
        period: TimeStepSpec,
        source: Optional[List[SourceBase]] = None,
        range: Optional[RangeSpec] = RangeSpec[:, :, :],  # default
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
        self.json = json if json is not None else {}
        self.json_restart = json_restart if json_restart is not None else {}
        self.data_preparation_strategy = data_preparation_strategy
        self.toml = toml
        self.particle_io_chunk_size = particle_io_chunk_size
        self.file_writing = file_writing
        self.check()

    def check(self):
        # particle_io_chunk_size must be positive
        if self.particle_io_chunk_size is not None and self.particle_io_chunk_size < 1:
            raise ValueError("particle_io_chunk_size (in MiB) must be positive")

        # infix must be NULL when using sst backend
        if self.ext == "sst" and self.infix is not None and self.infix != "NULL":
            raise ValueError("infix must be 'NULL' when ext is 'sst'")

        # validate sources
        if self.source is not None:
            if not all(isinstance(s, SourceBase) for s in self.source):
                raise ValueError("source must be a list of SourceBase objects")
            # validate species in sources
            for src in self.source:
                if hasattr(src, "species") and src.species is not None:
                    if not isinstance(src.species, PICMISpecies):
                        raise ValueError(f"Species {src.species} is not known to Simulation")

        # validate period
        if not isinstance(self.period, TimeStepSpec):
            raise TypeError("period must be a TimeStepSpec")
        for s in self.period.specs:
            if isinstance(s.step, (int, float)) and s.step < 1:
                raise ValueError("Step size must be >= 1")

        # validate range
        if not isinstance(self.range, RangeSpec):
            raise TypeError("range must be a RangeSpec")

    def get_as_pypicongpu(
        self,
        dict_species_picmi_to_pypicongpu: Dict,
        time_step_size: float,
        num_steps: int,
        simulation_box: Tuple[int, ...],
    ) -> PyPIConGPUOpenPMD:
        self.check()

        if len(simulation_box) != len(self.range):
            raise ValueError("Number of range specifications must match simulation box dimensions")

        sources = None
        if self.source is not None:
            sources = [src.get_as_pypicongpu(dict_species_picmi_to_pypicongpu) for src in self.source]

        pypicongpu_openpmd = PyPIConGPUOpenPMD(
            period=self.period.get_as_pypicongpu(time_step_size, num_steps),
            source=sources,
            range=self.range.get_as_pypicongpu(simulation_box),
            file=self.file,
            ext=self.ext,
            infix=self.infix,
            json=self.json,
            json_restart=self.json_restart,
            data_preparation_strategy=self.data_preparation_strategy,
            toml=self.toml,
            particle_io_chunk_size=self.particle_io_chunk_size,
            file_writing=self.file_writing,
        )
        return pypicongpu_openpmd
