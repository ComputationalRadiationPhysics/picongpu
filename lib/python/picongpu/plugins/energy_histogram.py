"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Axel Huebl
License: GPLv3+
"""

import numpy as np
import pandas as pd
import os
import collections


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
        return pd.read_csv(data_file_path,
                           skiprows=1,
                           usecols=(0,),
                           delimiter=" ",
                           dtype=np.uint64).as_matrix()[:, 0]

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
            A list of iterations is allowed as well.
            ``None`` refers to the list of all available iterations.
        include_overflow : boolean, default: False
            Include overflow and underflow bins as the first/last bins.

        Returns
        -------
        counts : np.array of dtype float [unitless]
            count of particles in each bin
            If iteration is a list, returns (ordered) dict with iterations as
            its index.
        bins : np.array of dtype float [keV]
            upper ranges of each energy bin
        """
        if iteration is not None:
            if not isinstance(iteration, collections.Iterable):
                iteration = np.array([iteration])

        data_file_path = self.get_data_path(species, species_filter)

        # read whole file as pandas.DataFrame
        data = pd.read_csv(
            data_file_path,
            skiprows=1,
            delimiter=" "
        )
        # upper range of each bin in keV
        #    note: only reads first row and selects the valid energy bins
        bins = pd.read_csv(
            data_file_path,
            comment=None,
            nrows=0,
            delimiter=" ",
            usecols=range(2, data.shape[1] - 2),
            dtype=np.float64
        ).columns.values.astype(np.float64)

        # set DataFrame column names properly
        data.columns = [
            'iteration',
            'underflow'
        ] + list(bins) + [
            'overflow',
            'sum'
        ]
        # set iteration as index
        data.set_index('iteration', inplace=True)

        # all iterations requested
        if iteration is None:
            iteration = np.array(data.index.values)

        # verify requested iterations exist
        if not set(iteration).issubset(data.index.values):
            raise IndexError('Iteration {} is not available!\n'
                             'List of available iterations: \n'
                             '{}'.format(iteration, data['iteration']))

        # remove unused columns
        del data['sum']
        if not include_overflow:
            del data['underflow']
            del data['overflow']

        if len(iteration) > 1:
            return collections.OrderedDict(zip(
                    iteration,
                    data.loc[iteration].as_matrix()
                )), bins
        else:
            return data.loc[iteration].as_matrix()[0, :], bins
