"""
This file is part of the PIConGPU.

Copyright 2017-2020 PIConGPU contributors
Authors: Axel Huebl
License: GPLv3+
"""
from .base_reader import DataReader

import collections
import numpy as np
import os
import glob
import re
import h5py as h5


class PhaseSpaceMeta(object):
    """
    Meta information for data of a phase space iteration.
    """

    def __init__(self, species, species_filter, ps, shape, extent, dV):
        """
        Parameters
        ----------
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)
        ps : string
            phase space selection in order: spatial, momentum component,
            e.g. 'ypy' or 'ypx'
        shape : 2-value tuple of int (r, p)
            shape of the phase space histogram in number of bins
        extent : length 4 np.array of floats (r_min, r_max, p_min, p_max)
            extent of the phase space histogram in SI
        dV : float
            spatial volume simulation cell in m
        """
        # strings for labels
        self.species = species
        self.species_filter = species_filter
        self.r = ps[0]
        self.p = ps[2]
        # lower and upper bound of bins in SI
        self.r_edges = np.linspace(extent[0], extent[1], shape[0] + 1)
        self.p_edges = np.linspace(extent[2], extent[3], shape[1] + 1)
        # ranges and conversions in SI
        self.extent = extent
        self.dV = dV
        self.dr = self.r_edges[1] - self.r_edges[0]
        self.dp = self.p_edges[1] - self.p_edges[0]


class PhaseSpaceData(DataReader):
    """
    Data Reader for the Phase Space Plugin.
    """

    def __init__(self, run_directory):
        """
        Parameters
        ----------
        run_directory : string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        """
        super().__init__(run_directory)

        self.data_file_prefix = "PhaseSpace_{0}_{1}_{2}_{3}"
        self.data_file_suffix = ".h5"
        self.data_hdf5_path = "/data/{0}/{1}"

    def get_data_path(self, ps, species, species_filter="all", iteration=None):
        """
        Return the path to the underlying data file.

        Parameters
        ----------
        ps : string
            phase space selection in order: spatial, momentum component,
            e.g. 'ypy' or 'ypx'
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)
        iteration : (unsigned) int or list of int [unitless]
            The iteration at which to read the data.
            If 'None', a regular expression string matching
            all iterations will be returned.

        Returns
        -------
        A string with a file path and a string with a in-file HDF5 path if
        iteration is a single value or a list of length one.
        If iteration is a list of length > 1, a list of paths is returned.
        If iteration is None, only the first string is returned and contains a
        regex-* for the position iteration.
        """
        if species is None:
            raise ValueError('The species parameter can not be None!')
        if species_filter is None:
            raise ValueError('The species_filter parameter can not be None!')
        if ps is None:
            raise ValueError('The ps parameter can not be None!')

        output_dir = os.path.join(
            self.run_directory,
            "simOutput",
            "phaseSpace"
        )
        if not os.path.isdir(output_dir):
            raise IOError('The simOutput/phaseSpace/ directory does not '
                          'exist inside path:\n  {}\n'
                          'Did you set the proper path to the '
                          'run directory?\n'
                          'Did you enable the phase space plugin?\n'
                          'Did the simulation already run?'
                          .format(self.run_directory))

        if iteration is not None:
            if not isinstance(iteration, collections.Iterable):
                iteration = [iteration]

            ret = []
            for it in iteration:
                data_file_name = self.data_file_prefix.format(
                    species,
                    species_filter,
                    ps,
                    str(it)) + self.data_file_suffix
                data_file_path = os.path.join(output_dir, data_file_name)

                if not os.path.isfile(data_file_path):
                    raise IOError('The file {} does not exist.\n'
                                  'Did the simulation already run?'
                                  .format(data_file_path))

                data_hdf5_name = self.data_hdf5_path.format(
                    it,
                    ps)

                ret.append((data_file_path, data_hdf5_name))
            if len(iteration) == 1:
                return ret[0]
            else:
                return ret
        else:
            iteration_str = "*"

            data_file_name = self.data_file_prefix.format(
                species,
                species_filter,
                ps,
                iteration_str
            ) + self.data_file_suffix
            return os.path.join(output_dir, data_file_name)

    def get_iterations(self, ps, species, species_filter='all'):
        """
        Return an array of iterations with available data.

        Parameters
        ----------
        ps : string
            phase space selection in order: spatial, momentum component,
            e.g. 'ypy' or 'ypx'
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)

        Returns
        -------
        An array with unsigned integers.
        """
        # get the regular expression matching all available files
        data_file_path = self.get_data_path(ps, species, species_filter)

        matching_files = glob.glob(data_file_path)
        re_it = re.compile(data_file_path.replace("*", "([0-9]+)"))

        iterations = np.array(
            sorted(
                map(
                    lambda file_path:
                    np.uint64(re_it.match(file_path).group(1)),
                    matching_files
                )
            ),
            dtype=np.uint64
        )

        return iterations

    def _get_for_iteration(self, iteration, ps, species, species_filter='all',
                           **kwargs):
        """
        Get a phase space histogram.

        Parameters
        ----------
        iteration : (unsigned) int [unitless] or list of int or None.
            The iteration at which to read the data.
            ``None`` refers to the list of all available iterations.
        ps : string
            phase space selection in order: spatial, momentum component,
            e.g. 'ypy' or 'ypx'
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)

        Returns
        -------
        ps : np.ndarray of dtype float, shape(nr, np) [...]
            ...
        ps_meta :
            PhaseSpaceMeta object with meta information about the 2D histogram

        If iteration is a list (or None), return a list of tuples
        containing ps and ps_meta for each requested iteration.
        If a single iteration is requested, return the tuple (ps, ps_meta).
        """
        available_iterations = self.get_iterations(
            ps, species, species_filter)

        if iteration is not None:
            if not isinstance(iteration, collections.Iterable):
                iteration = [iteration]
            # verify requested iterations exist
            if not set(iteration).issubset(available_iterations):
                raise IndexError('Iteration {} is not available!\n'
                                 'List of available iterations: \n'
                                 '{}'.format(iteration, available_iterations))
        else:
            # take all availble iterations
            iteration = available_iterations

        ret = []
        for it in iteration:
            data_file_path, data_hdf5_name = self.get_data_path(
                ps,
                species,
                species_filter,
                it)

            f = h5.File(data_file_path, 'r')
            ps_data = f[data_hdf5_name]

            # all in SI
            dV = ps_data.attrs['dV'] * ps_data.attrs['dr_unit']**3
            unitSI = ps_data.attrs['sim_unit']
            p_range = ps_data.attrs['p_unit'] * \
                np.array([ps_data.attrs['p_min'], ps_data.attrs['p_max']])

            mv_start = ps_data.attrs['movingWindowOffset']
            mv_end = mv_start + ps_data.attrs['movingWindowSize']
            #                2D histogram:         0 (r_i); 1 (p_i)
            spatial_offset = ps_data.attrs['_global_start'][1]

            dr = ps_data.attrs['dr'] * ps_data.attrs['dr_unit']

            r_range_cells = np.array([mv_start, mv_end]) + spatial_offset
            r_range = r_range_cells * dr

            extent = np.append(r_range, p_range)

            # cut out the current window & scale by unitSI
            ps_cut = ps_data[mv_start:mv_end, :] * unitSI

            f.close()

            ps_meta = PhaseSpaceMeta(
                species, species_filter, ps, ps_cut.shape, extent, dV)
            ret.append((ps_cut, ps_meta))

        if len(iteration) == 1:
            return ret[0]
        else:
            return ret
