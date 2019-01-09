"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sophie Rudat
License: GPLv3+
"""
from picongpu.plugins.data import EmittanceData
from picongpu.plugins.data import EnergyHistogramData
from .base_reader import DataReader

import numpy as np
import os
import collections
import json


class CompareData(DataReader):
    """
    Data Reader for showing the variation of ?? (TODO).
    """

    def __init__(self, run_directory):
        """
        Parameters
        ----------
        run_directory : string
            path to a scan directory which contains a 'scan_ranges.json'
            file and one or more picongpu simulation directories.
        """
        super().__init__(run_directory)
        self.json_suffix = "scan_ranges.json"

    def _get_all_sim_paths(self):
        """
        List all simulation directories within the scan directory
        """
        sim_dirs = [d for d in os.listdir(self.run_directory)
                    if os.path.isdir(d)]

        sims_rundir = []
        for dir_name in sim_dirs:
            sim_path = os.path.join(self.run_directory, dir_name, "run")
            if os.path.isdir(sim_path):
                sims_rundir.append(sim_path)

        return sims_rundir

    def get_data_path(self, species, variation,
                      species_filter="all"):
        """
        Return the path to the underlying data file.

        Parameters
        ----------
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        variation: str
            name of the varied input parameter
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)

        Returns
        -------
        A list of strings with path to all sims in scan and a
        list with the values for the choosen variation.
        """
        # get species, filter
        if species is None:
            raise ValueError('The species parameter can not be None!')
        if species_filter is None:
            raise ValueError('The species_filter parameter can not be None!')
        if variation is None:
            raise ValueError('The variation parameter can not be None!')

        path_to_json = os.path.join(self.run_directory, self.json_suffix)
        if not os.path.isfile(path_to_json):
            raise IOError('The file {} does not exist.\n'
                          'Did the simulation already run?'
                          .format(path_to_json))

        with open(path_to_json) as f:
            range_dict = json.load(f)
            filtered_range_dict = dict()
            for name, attrs in range_dict.items():

                filtered_range_dict[name] = attrs['values']
                # TODO: this could be a list of discrete options or a list of
                # triples for range based parameters which we have to discern

        sims_rundir = self._get_all_sim_paths()

        return sims_rundir, filtered_range_dict[variation]

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

        # TODO: why EnergyHistogram?
        #       run_directory points to a scan and not a specific simulation
        #       so where do we get the information from? Intersection of all
        #       simulations of the scan? But maybe different plugins have
        #       different iterations present so we can't rely on the
        #       EnergyHistogram.
        #       Should we just leave it unimplemented?
        iteration = EnergyHistogramData(
                    self.run_directory).get_iterations(species=species)
        return iteration

    def _get_for_iteration(self, iteration, species, variation,
                           observable, species_filter="all", **kwargs):
        """

        Parameters
        ----------
        iteration : (unsigned) int [unitless]
            The iteration at which to read the data.
            A list of iterations is allowed as well.
            ``None`` refers to the list of all available iterations.
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        variation : string
            name of the variated input parameter
        observable: string
            name of the calculated and compared quantity.
            One of ["emittance", "bunchEnergy",
                    "energySpread", "energySpread/bunchEnergy",
                    "maxEnergy", "charge"]
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)

        Returns
        -------
        params: ??
            ...
        values: ??
            ...
        variation : np.array
            values of the variated input parameter
        observables : np.array
            calculated observed values for the respectiv input parameter
        other_params: dict
            values of other variated input parameters
        """
        # get list of path to files
        if iteration is not None:
            if not isinstance(iteration, collections.Iterable):
                iteration = np.array([iteration])

        sims_rundir, params = self.get_data_path(species, variation,
                                                 observable,
                                                 species_filter)
        values = []
        params = []
        for sim in (sims_rundir):
            # was path to 'run' subdir, but json is one directory above
            sim_json = os.path.join(os.path.dirname(sim), 'params.json')
            if not os.path.isfile(sim_json):
                raise IOError('The file {} does not exist.'.format(sim_json))
            with open(sim_json) as f:
                range_dict = json.load(f)
                param = range_dict[variation]['values'][0]
            try:
                if observable == "emittance":
                    counts, bins, _ = EmittanceData(sim).get(
                        species, species_filter)
                    emit = counts[iteration[0]][0] * 1.e6
                    value = emit  # unit: pi mm mrad

                else:
                    counts, bins, _, _ = EnergyHistogramData(sim).get(
                                   iteration=iteration, species=species)
                    # TODO: comment on magic numbers
                    start = 80
                    limitE = 500
                    data = counts[start:]
                    for i in np.arange(data.size):
                        if (max(data) == data[i]):
                            peakEnergy = i
                    for j in np.arange(data.size-peakEnergy):
                        if (max(data)/3 - data[peakEnergy + j] > 0.0):
                            FWHM_max = ((peakEnergy + j + start) *
                                        limitE / len(bins))
                            break
                    for k in np.arange(peakEnergy):
                        if (max(data)/3-data[peakEnergy - k] > 0.0):
                            FWHM_min = ((peakEnergy - k + start)
                                        * limitE / len(bins))
                            break
                    if observable == "bunchEnergy":
                        maxEnergy = 0.5 * (FWHM_max + FWHM_min)
                        value = maxEnergy  # unit: MeV
                    elif observable == "energySpread":
                        energySpread = (FWHM_max - FWHM_min)
                        value = energySpread  # unit: MeV
                    elif observable == "energySpread/bunchEnergy":
                        energySpread = (FWHM_max - FWHM_min)
                        maxEnergy = 0.5 * (FWHM_max + FWHM_min)
                        value = energySpread/maxEnergy
                    elif observable == "maxEnergy":
                        csum = np.cumsum(data[::-1])
                        # 6.25e6 = 1pC
                        maxE = ((start + np.shape(
                                data[[csum > 6.25e6][::-1]])[0])
                                * limitE / len(bins))
                        value = maxE   # unit: MeV
                    elif observable == "charge":
                        for j in np.arange(data.size-peakEnergy):
                            if (max(data) / 3 - data[peakEnergy + j] > 0.0):
                                cmax = (peakEnergy + j)
                                break
                        for k in np.arange(peakEnergy):
                            if (max(data) / 3 - data[peakEnergy - k] > 0.0):
                                cmin = (peakEnergy - k)
                                break
                        # charge within +/- 1/3 peakEnergy, unit pC
                        charge = (data[cmin:cmax].sum() *
                                  1.6021766209e-19 * 1.e12)
                        value = charge
                    else:
                        raise ValueError("Unknown observable {}".format(
                            observable))

            except (ValueError, IOError, IndexError) as e:
                print("Error in sim ", sim, ":", e)
                value = np.nan

            values.append(value)
            params.append(param)
        return params, values
