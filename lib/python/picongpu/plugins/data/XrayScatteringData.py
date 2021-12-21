"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Pawel Ordyna
License: GPLv3+
"""
from .base_reader import DataReader

from os import path
import numpy as np
import openpmd_api as api


class XrayScatteringData(DataReader):
    """ Data reader for the xrayScattering plugin. """

    def __init__(self, run_directory, species, file_extension='bp',
                 file_name_base='Output'):
        """
        Parameters
        ----------
        run_directory : string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        species : string
            Species for which the plugin output should be loaded. It's the
            string defined in `speciesDefinition.param`.
        file_extension : string
            file extension of the xrayScattering output file.
            Default is "bp".
        file_name_base : string
            String name set in the xrayScattering command line parameter
            fileName. Default is "Output".
            The full file name is

            ::
                `<species>_xrayScattering<file_name_base>.<file_extension>`
        """

        super().__init__(run_directory)

        self.full_file_name = (species + "_xrayScattering" + file_name_base +
                               "." + file_extension)

        self.full_path = path.join(self.run_directory,
                                   "simOutput/xrayScatteringOutput")
        self.full_path = path.join(self.full_path, self.full_file_name)
        # openPMD series
        self.series = api.Series(self.full_path, api.Access_Type.read_only)
        self.total_simulation_cells = self.series.get_attribute(
            "totalSimulationCells")

    def get_data_path(self, **kwargs):
        """
        Returns
        -------
        A string with the path to the underlying data file.
        """
        return self.full_path

    def get_iterations(self, **kwargs):
        """
        Returns
        -------
        An array with unsigned integers of iterations for which
        data is available.
        """
        return np.array(list(self.series.iterations))

    def _get_for_iteration(self, iteration, **kwargs):
        """ Get the data for a given iteration in PIC units.

        Call `get_unit` method to get the conversion factor (to SI).

        Returns
        -------
        The complex scattering amplitude in PIC units.
        """

        i = self.series.iterations[iteration]
        amplitude = i.meshes['amplitude']
        mrc_real, mrc_imag = amplitude['x'], amplitude['y']
        real = mrc_real.load_chunk()
        imag = mrc_imag.load_chunk()
        self.series.flush()
        if mrc_imag.dtype.type is np.float32:
            dtype = np.complex64
        elif mrc_imag.dtype.type is np.float64:
            dtype = np.complex128
        else:
            raise TypeError
        result = (real + 1j * imag) * self.total_simulation_cells
        return result.astype(dtype)

    def get_unit(self):
        """ Get the amplitude unit. """
        i = self.series.iterations[self.get_iterations()[0]]
        amplitude = i.meshes['amplitude']
        mrc_real = amplitude['x']
        return mrc_real.unit_SI
