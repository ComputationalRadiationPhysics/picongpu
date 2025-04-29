"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
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
    Specifies parameters for openPMD diagnostic output in particle-in-cell simulations.

    This diagnostic outputs simulation data (fields and particles) to disk using the openPMD
    standard, with configurable periods, data sources, and backend settings.

    Attention: **period is mandatory.**

    Parameters
    ----------
    period: TimeStepSpec
        Specifies the time steps for data output.
        Unit: simulation time steps. Required.

    source: List[SourceBase], optional
        List of data source objects to dump (e.g., [ChargeDensity(filter="all")]).
        Default: None (no sources specified).

    range: str, optional
        Contiguous range of cells per dimension to dump (e.g., ":,:,:").
        Format: comma-separated ranges like "begin:end,begin:end,begin:end".
        Default: ":,::,:" (all cells).

    file: str, optional
        Relative or absolute file prefix for openPMD output files.
        If relative, files are stored under the simulation output directory.
        Default: None (backend-dependent default).

    ext: Literal["bp", "h5", "sst"], optional
        File extension controlling the openPMD backend.
        Options: "bp" (ADIOS2), "h5" (HDF5), "sst" (ADIOS2/SST for streaming).
        Default: "bp" (ADIOS2 backend).

    infix: str, optional
        Filename infix for iteration layout (e.g., "_%06T").
        Use "NULL" for group-based layout. Required as "NULL" if ext="sst".
        Default: "NULL" (group-based layout).

    json: Union[str, Dict], optional
        Backend-specific parameters for writing, as a JSON string, dictionary, or filename
        (filename must be prepended with "@").
        Default: {} (empty dictionary).

    json_restart: Union[str, Dict], optional
        Backend-specific parameters for restarting, as a JSON string, dictionary, or filename
        (filename must be prepended with "@").
        Default: {} (empty dictionary).

    data_preparation_strategy: Literal["doubleBuffer", "adios", "mappedMemory", "hdf5"], optional
        Strategy for particle data preparation.
        Options: "doubleBuffer" or "adios" (ADIOS2-based), "mappedMemory" or "hdf5" (HDF5-based).
        Default: None (backend-dependent).

    toml: str, optional
        Path to a TOML file for openPMD configuration.
        Default: None.

    particle_io_chunk_size: int, optional
        Size of particle data chunks for writing (in MiB).
        Reduces host memory footprint for certain backends.
        Default: None (backend-dependent).

    file_access: Literal["create", "append"], optional
        File access mode for writing.
        Options: "create" (new files), "append" (for checkpoint-restart workflows).
        Default: "create" (new files).
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
