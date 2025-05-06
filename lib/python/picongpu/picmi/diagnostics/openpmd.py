"""
This file is part of PIConGPU.
Copyright 2025-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.openpmd import OpenPMD as PyPIConGPUOpenPMD
from ...pypicongpu.output import Source as PyPIConGPUSource
from .timestepspec import TimeStepSpec

from .openpmd_sources.source_base import SourceBase

import typeguard
from typing import Optional, Dict, Union, List, Literal


@typeguard.typechecked
class OpenPMD:
    """
    openPMD diagnostic output

    This diagnostic writes simulation data (base fields, derived fields and/or particles) to disk using the openPMD
    standard, with configurable periods, data sources and backend settings.

    @param period specification of the time steps for data output, outputs will always be written at the end of a PIC time step.
    @param source list of data source objects to include in the dump (e.g., [ChargeDensity(filter="all")]),
        Setting this to None will cause an empty dump
    @param range contiguous range of cells to dump the base- and derived field for
        specification as comma-separated "begin:end" range for each dimension, i.e. "begin:end,begin:end,begin:end"
        Notes: Values will be clipped to the simulation box. Begin and/or end may be omitted to indicate the limit of the simulation box in the dimension. The Default value ":,::,:" indicates that fields should be dumped for all cells.
    @param file relative or absolute file path prefix for openPMD output files. Relative paths are interpreted as relative to the simulation output directory, the default value None indicates the PIC code's default.
    @param ext file extension controlling the openPMD backend, options are "bp" (default backend ADIOS2), "bp4" (bp4 backend ADIOS2), "bp5" (bp5 backend ADIOS2), "h5" (HDF5), "sst" (ADIOS2/SST for streaming).
    @param infix filename infix for the iteration layout (e.g., "_%06T"), use "NULL" for the group-based layout, ext="sst" requires infix="NULL".
    @param json openPMD backend configuration as a JSON string, dictionary, or filename (filename must be prepended with "@").
    @param json_restart backend-specific parameters for restarting, as a JSON string, dictionary, or filename (filenames must be prepended with "@").
    @param data_preparation_strategy strategy for particle data preparation, options: "doubleBuffer" or "adios" (ADIOS2-based), "mappedMemory" or "hdf5" (HDF5-based), the default value None indicates the PIC code default
    @param toml path to a TOML file for openPMD configuration. Replaces the JSON or keyword configuration.
    @param particle_io_chunk_size size of particle data chunks used in writing (in MiB), reduces host memory footprint for certain backends, default "None" indicates the PIC code default.
    @param file_access file access mode for writing, options: "create" (new files), "append" (for checkpoint-restart workflows).
    """

    def check(self):
        """
        Validate the provided parameters.
        """
        if self.period is None:
            raise ValueError("period is mandatory")
        if self.particle_io_chunk_size is not None and self.particle_io_chunk_size < 1:
            raise ValueError("particle_io_chunk_size (in MiB) must be positive")
        if self.ext == "sst" and self.infix is not None and self.infix != "NULL":
            raise ValueError("infix must be 'NULL' when ext is 'sst'")
        if self.source is not None and not all(isinstance(s, SourceBase) for s in self.source):
            raise ValueError("source must be a list of SourceBase objects")

    def __init__(
        self,
        period: TimeStepSpec,
        source: Optional[List[SourceBase]] = None,
        range: Optional[str] = ":,:,:",
        file: Optional[str] = None,
        ext: Optional[Literal["bp", "h5", "sst"]] = "bp",
        infix: Optional[str] = "NULL",
        json: Optional[Union[str, Dict]] = None,
        json_restart: Optional[Union[str, Dict]] = None,
        data_preparation_strategy: Optional[Literal["doubleBuffer", "adios", "mappedMemory", "hdf5"]] = None,
        toml: Optional[str] = None,
        particle_io_chunk_size: Optional[int] = None,
        file_access: Optional[Literal["create", "append"]] = "create",
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
        self.file_access = file_access

        self.check()

    def get_as_pypicongpu(
        self,
        pypicongpu_by_picmi_species: Dict,
        time_step_size: float,
        num_steps: int,
    ) -> PyPIConGPUOpenPMD:
        self.check()

        pypicongpu_openpmd = PyPIConGPUOpenPMD(
            period=self.period.get_as_pypicongpu(time_step_size, num_steps),
            source=PyPIConGPUSource([s.get_as_pypicongpu() for s in self.source]) if self.source is not None else None,
            range=self.range,
            file=self.file,
            ext=self.ext,
            infix=self.infix,
            json=self.json,
            json_restart=self.json_restart,
            data_preparation_strategy=self.data_preparation_strategy,
            toml=self.toml,
            particle_io_chunk_size=self.particle_io_chunk_size,
            file_access=self.file_access,
        )
        return pypicongpu_openpmd
