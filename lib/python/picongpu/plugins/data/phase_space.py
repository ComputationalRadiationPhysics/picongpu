"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Axel Huebl
License: GPLv3+
"""
from .base_reader import DataReader

import collections
import numpy as np
import os
import openpmd_api as io


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
        self.data_hdf5_path = "/data/{0}/{1}"

    def get_data_path(self, ps, species, species_filter="all", file_ext="h5"):
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
        file_ext: string
            filename extension for openPMD backend
            default is 'h5' for the HDF5 backend

        Returns
        -------
        A string with a the full openPMD file path pattern for loading from
        a file-based iteration layout.
        """
        # @todo different file extensions?
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

        iteration_str = "%T"
        data_file_name = self.data_file_prefix.format(
            species,
            species_filter,
            ps,
            iteration_str
        ) + '.' + file_ext
        return os.path.join(output_dir, data_file_name)

    def get_iterations(self, ps, species, species_filter='all',
                       file_ext="h5"):
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
        file_ext: string
            filename extension for openPMD backend
            default is 'h5' for the HDF5 backend

        Returns
        -------
        An array with unsigned integers.
        """
        # get the regular expression matching all available files
        data_file_path = self.get_data_path(ps, species, species_filter,
                                            file_ext=file_ext)

        series = io.Series(data_file_path, io.Access.read_only)
        iterations = [key for key, _ in series.iterations.items()]

        return iterations

    def _get_for_iteration(self, iteration, ps, species, species_filter='all',
                           file_ext="h5", **kwargs):
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
        file_ext: string
            filename extension for openPMD backend
            default is 'h5' for the HDF5 backend

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

        data_file_path = self.get_data_path(ps, species, species_filter,
                                            file_ext=file_ext)
        series = io.Series(data_file_path, io.Access.read_only)
        available_iterations = [key for key, _ in series.iterations.items()]

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
        for index in iteration:
            it = series.iterations[index]
            dataset_name = "{}_{}_{}".format(species, species_filter, ps)
            mesh = it.meshes[dataset_name]
            ps_data = mesh[io.Mesh_Record_Component.SCALAR]

            # all in SI
            dV = mesh.get_attribute('dV') * mesh.get_attribute('dr')**3
            unitSI = mesh.get_attribute('sim_unit')
            p_range = mesh.get_attribute('p_unit') * \
                np.array(
                    [mesh.get_attribute('p_min'), mesh.get_attribute('p_max')])

            mv_start = mesh.get_attribute('movingWindowOffset')
            mv_end = mv_start + mesh.get_attribute('movingWindowSize')
            #                2D histogram:         0 (r_i); 1 (p_i)
            spatial_offset = mesh.get_attribute('_global_start')[0]

            dr = mesh.get_attribute('dr') * mesh.get_attribute('dr_unit')

            r_range_cells = np.array([mv_start, mv_end]) + spatial_offset
            r_range = r_range_cells * dr

            extent = np.append(r_range, p_range)

            # cut out the current window & scale by unitSI
            ps_cut = ps_data[mv_start:mv_end, :] * unitSI

            it.close()

            ps_meta = PhaseSpaceMeta(
                species, species_filter, ps, ps_cut.shape, extent, dV)
            ret.append((ps_cut, ps_meta))

        if len(iteration) == 1:
            return ret[0]
        else:
            return ret
