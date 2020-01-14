"""
This file is part of the PIConGPU.

Copyright 2017-2020 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""
from ...utils.find_time import FindTime

import numpy as np


class DataReader(object):
    """
    Base class that all data readers should inherit from.
    """
    def __init__(self, run_directory):
        """
        Parameters
        ----------
        run_directory: string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        """
        if run_directory is None:
            raise ValueError('The run_directory parameter can not be None!')

        self.run_directory = run_directory
        self.find_time = FindTime(run_directory)

        # need to be set in derived classes
        self.data_file_prefix = None
        self.data_file_suffix = None

    def get_dt(self):
        """
        Return the timestep for the chosen simulation.
        """
        return self.find_time.get_dt()

    def get_times(self, *args, **kwargs):
        """
        Returns
        -------
        An array of floats of simulation time steps for which
        data is available
        """

        iterations = np.array(self.get_iterations(*args, **kwargs))
        return self.find_time.get_time(iterations)

    def get_data_path(self, *args, **kwargs):
        """
        Returns
        -------
        A string with the path to the underlying data file.
        """
        raise NotImplementedError

    def get_iterations(self, *args, **kwargs):
        """
        Returns
        -------
        An array with unsigned integers of iterations for which
        data is available.
        """
        raise NotImplementedError

    def get(self, *args, **kwargs):
        """
        Parameters
        ----------
        Either 'iteration' or 'time' should be present in the kwargs.
        If both are given, the 'time' argument is converted to
        an iteration and data for the iteration matching the time
        is returned.
        For other valid args and kwargs, please look at the
        documentation of the '_get_for_iteration' methods
        of the derived classes since the parameters are passed
        on to that function.

        time: float or np.array of float or None.
            If None, data for all available times is returned.

        iteration: int or np.array of int or None.
            If None, data for all available iterations is returned.

        Returns
        -------
        The data for the requested parameters in a plugin
        dependent format and type.
        """
        if 'iteration' not in kwargs and 'time' not in kwargs:
            raise ValueError(
                "One of 'iteration' and 'time' parameters"
                " has to be present!")

        iteration = None
        if 'iteration' in kwargs:
            # remove the entry from kwargs since we pass that
            # on
            iteration = kwargs.pop('iteration')

        if 'time' in kwargs:
            # we have time and override the iteration
            # by the converted time
            time = kwargs.pop('time')
            if time is None:
                # use all times that are available, i.e. all iterations
                iteration = self.get_iterations(*args, **kwargs)
            else:
                iteration = self.find_time.get_iteration(
                    time, method='closest')
            # print("got 'time'=", time, ", converted to iter", iteration)

        return self._get_for_iteration(iteration, *args, **kwargs)

    def _get_for_iteration(self, iteration, *args, **kwargs):
        """
        Get the data for a given iteration.

        Returns
        -------
        The data for the requested parameters in a plugin
        dependent format and type.
        """
        raise NotImplementedError
