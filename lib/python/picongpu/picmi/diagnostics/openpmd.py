"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Masoud Afshari
License: GPLv3+
"""

from ...pypicongpu.output.openpmd import OpenPMD as PyPIConGPUOpenPMD
from .timestepspec import TimeStepSpec
from .openpmd_sources.source import Source

import typeguard
from typing import Optional, Dict, Union


@typeguard.typechecked
class OpenPMD:
    """
    Specifies the parameters for the openPMD plugin in PIConGPU simulations.

    This plugin outputs simulation data (fields and particles) to disk using the openPMD API,
    with configurable periods, data sources, and backend settings.

    Attention: **period is mandatory.**

    Parameters
    ----------
    period: TimeStepSpec
        Specify on which time steps to output data.
        Unit: steps (simulation time steps). Required.

    source: Source, optional
        Data sources and filters to dump (e.g., Source([ChargeDensity(filter="filterX"), "species_all"])).
        Default: Source(["species_all", "fields_all"]).

    range: str, optional
        Contiguous range of cells per dimension to dump (e.g., ":,:,:").
        Format: comma-separated ranges like "begin:end,begin:end,begin:end".
        Default: ":,:,:"

    file: str, optional
        Relative or absolute file prefix for openPMD output files.
        If relative, files are stored under simOutput.

    ext: str, optional
        File extension controlling the openPMD backend.
        Options: "bp" (ADIOS2, default), "h5" (HDF5), "sst" (ADIOS2/SST) for data streaming.

    infix: str, optional
        Filename infix for iteration layout (e.g., "_%06T").
        Set to "NULL" for group-based layout. Mandatory "NULL" if ext="sst" for data streaming.
        Default: "_%06T".

    json: Union[str, Dict], optional
        Backend-specific parameters for writing, as a JSON string, dictionary, or filename
        (filename must be prepended with "@"). Default: empty dictionary.

    json_restart: Union[str, Dict], optional
        Backend-specific parameters for restarting, as a JSON string, dictionary, or filename
        (filename must be prepended with "@"). Default: empty dictionary.

    data_preparation_strategy: str, optional
        Strategy for particle data preparation.
        Options: "doubleBuffer" (alias "adios", default), "mappedMemory" (alias "hdf5").

    toml: str, optional
        Path to a TOML file for openPMD configuration.

    particle_io_chunk_size: int, optional
        Size of particle data chunks for writing (in MiB).
        Reduces host memory footprint for bp5 backend.

    file_access: str, optional
        File access mode for writing.
        Options: "create" (default), "append" (for checkpoint-restart workflows).
    """

    def check(self):
        """
        Validate the provided parameters.
        """
        if self.period is None:
            raise ValueError("period is mandatory")
        if self.ext is not None and self.ext not in ["bp", "h5", "sst"]:
            raise ValueError("ext must be one of 'bp', 'h5', 'sst'")
        if self.data_preparation_strategy is not None and self.data_preparation_strategy not in [
            "doubleBuffer",
            "adios",
            "mappedMemory",
            "hdf5",
        ]:
            raise ValueError("data_preparation_strategy must be one of 'doubleBuffer', 'adios', 'mappedMemory', 'hdf5'")
        if self.file_access is not None and self.file_access not in ["create", "append"]:
            raise ValueError("file_access must be one of 'create', 'append'")
        if self.particle_io_chunk_size is not None and self.particle_io_chunk_size < 1:
            raise ValueError("particle_io_chunk_size (in MiB) must be positive")
        if self.ext == "sst" and self.infix is not None and self.infix != "NULL":
            raise ValueError("infix must be 'NULL' when ext is 'sst'")
        if not isinstance(self.source, Source):
            raise ValueError("source must be a Source object")

    def __init__(
        self,
        period: TimeStepSpec,
        source: Optional[Source] = None,
        range: Optional[str] = None,
        file: Optional[str] = None,
        ext: Optional[str] = None,
        infix: Optional[str] = None,
        json: Optional[Union[str, Dict]] = None,
        json_restart: Optional[Union[str, Dict]] = None,
        data_preparation_strategy: Optional[str] = None,
        toml: Optional[str] = None,
        particle_io_chunk_size: Optional[int] = None,
        file_access: Optional[str] = None,
    ):
        self.period = period
        self.source = source if source is not None else Source(["species_all", "fields_all"])
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

        pypicongpu_openpmd = PyPIConGPUOpenPMD()
        pypicongpu_openpmd.period = self.period.get_as_pypicongpu(time_step_size, num_steps)
        pypicongpu_openpmd.source = self.source.get_as_pypicongpu()
        pypicongpu_openpmd.range = self.range
        pypicongpu_openpmd.file = self.file
        pypicongpu_openpmd.ext = self.ext
        pypicongpu_openpmd.infix = self.infix
        pypicongpu_openpmd.json = self.json
        pypicongpu_openpmd.json_restart = self.json_restart
        pypicongpu_openpmd.data_preparation_strategy = self.data_preparation_strategy
        pypicongpu_openpmd.toml = self.toml
        pypicongpu_openpmd.particle_io_chunk_size = self.particle_io_chunk_size
        pypicongpu_openpmd.file_access = self.file_access

        return pypicongpu_openpmd
