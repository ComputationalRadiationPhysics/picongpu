"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Axel Huebl
License: GPLv3+
"""

import numpy as np
import os


class EnergyHistogram(object):
    """
    Data Reader for the Energy Histogram Plugin.
    """

    def __init__(self, run_directory):
        """
        Parameters
        ----------
        simulation_directory : string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        """
        if run_directory is None:
            raise ValueError('The run_directory parameter can not be None!')

        self.run_directory = run_directory
        self.data_file_prefix = "_energyHistogram_"
        self.data_file_suffix = ".dat"

    def get_data_path(self, species, species_filter="all"):
        """
        Return the path to the underlying data file.

        Parameters
        ----------
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)

        Returns
        -------
        A string with a path.
        """
        if species is None:
            raise ValueError('The species parameter can not be None!')
        if species_filter is None:
            raise ValueError('The species_filter parameter can not be None!')

        sim_output_dir = os.path.join(self.run_directory, "simOutput")
        if not os.path.isdir(sim_output_dir):
            raise IOError('The simOutput/ directory does not exist inside '
                          'path:\n  {}\n'
                          'Did you set the proper path to the run directory?\n'
                          'Did the simulation already run?'
                          .format(self.run_directory))

        data_file_path = os.path.join(
            sim_output_dir,
            species + self.data_file_prefix + species_filter +
            self.data_file_suffix
        )
        if not os.path.isfile(data_file_path):
            raise IOError('The file {} does not exist.\n'
                          'Did the simulation already run?'
                          .format(data_file_path))

        return data_file_path

    def get_iterations(self, species, species_filter="all"):
        """
        Return an array of iterations with available data.

        Parameters
        ----------
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
        data_file_path = self.get_data_path(species, species_filter)

        # the first column contains the iterations
        return np.loadtxt(data_file_path,
                          usecols=(0,),
                          dtype=np.uint64)

    def get(self, species, species_filter="all", iteration=None,
            include_overflow=False):
        """
        Get a histogram.

        Parameters
        ----------
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)
        iteration : (unsigned) int [unitless]
            The iteration at which to read the data.
            @TODO also allow lists here
        include_overflow : boolean, default: False
            Include overflow and underflow bins as the first/last bins.

        Returns
        -------
        energies : np.array of dtype float [unitless]
            count of particles in each bin
            @todo if iteration is a list, return a dict
        bins : np.array of dtype float [keV]
            upper ranges of each energy bin
        """
        if iteration is None:
            raise ValueError('The iteration needs to be set!')

        data_file_path = self.get_data_path(species, species_filter)

        # matrix with data in keV
        #   note: skips columns for iteration step, underflow bin (first 2) and
        #         overflow bin, summation of all bins (last two)
        ene = np.loadtxt(data_file_path)[:, 2:-2]
        # upper range of each bin in keV
        #    note: only reads first row and selects the actual valid bins
        #    (default, see self.num_bins) in columns 2:num_bins+2
        num_bins = ene.shape[1]
        extra_bins = 0
        if include_overflow:
            extra_bins = 1
        bins = np.loadtxt(
            data_file_path,
            comments=None,
            usecols=list(range(2 - extra_bins, num_bins + 2 + 2 * extra_bins))
        )[0, :]

        available_iterations = self.get_iterations(species, species_filter)
        if iteration not in available_iterations:
            raise IndexError('Iteration {} is not available!\n'
                             'List of available iterations: \n'
                             '{}'.format(iteration, available_iterations))

        dump_index = np.where(available_iterations == iteration)[0][0]

        return ene[dump_index], bins
