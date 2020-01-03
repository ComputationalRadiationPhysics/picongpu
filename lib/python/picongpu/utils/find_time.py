"""
This file is part of the PIConGPU.

Copyright 2017-2020 PIConGPU contributors
Authors: Axel Huebl
License: GPLv3+
"""

import numpy as np
import os


class FindTime(object):
    """
    Convert iterations (time steps) to time [seconds] and back.
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
        self.data_file = "output"

        self.dt = self.get_dt()

    def get_data_path(self):
        """
        Return the path to the underlying data file.

        Returns
        -------
        A string with a path.
        """
        sim_output_dir = os.path.join(self.run_directory, "simOutput")
        if not os.path.isdir(sim_output_dir):
            raise IOError('The simOutput/ directory does not exist inside '
                          'path:\n  {}\n'
                          'Did you set the proper path to the run directory?\n'
                          'Did the simulation already run?'
                          .format(self.run_directory))

        data_file_path = os.path.join(sim_output_dir, self.data_file)
        if not os.path.isfile(data_file_path):
            raise IOError('The file {} does not exist.\n'
                          'Did the simulation already run?'
                          .format(data_file_path))

        return data_file_path

    def get_dt(self):
        """
        Returns the time step of the simulation.

        Returns
        -------
        A float in seconds.
        """
        data_file_path = self.get_data_path()

        # matches floats and scientific floats
        rg_flt = r"([-\+[0-9]+\.[0-9]*[Ee]*[\+-]*[0-9]*)"

        # our UNIT_TIME is scaled to dt
        dt = np.fromregex(
            data_file_path,
            r"\s+UNIT_TIME " +
            rg_flt + r"\n",
            dtype=np.dtype([('dt', 'float')])
        )

        return dt['dt'][0]

    def get_time(self, iteration):
        """
        Find a time in seconds for a given iteration.

        Parameters
        ----------
        iteration : (unsigned) integer
            an iteration

        Returns
        -------
        time: float in seconds
            a matching time
        """
        return iteration * self.dt

    def get_iteration(self, t, iterations=None, method='previous'):
        """
        Find an iteration for a given time in seconds.

        Parameters
        ----------
        t : float
            time in seconds
        iterations : np.array of dtype np.uint64
            an array of iterations to choose from.
            defaults to full integers
        methods : string
            The method how to find a matching iteration.
            previous : the closest iteration that is <= t (default)
            closest : the closest iteration to t
            next : the next iteration that is > t

        Returns
        -------
        iteration : (unsigned) integer
            a matching iteration
        new_time : float
            the time at iteration in seconds
        """
        if t is None:
            raise ValueError('The time t needs to be set!')

        implemented_methods = ['previous', 'closest', 'next']
        if method not in implemented_methods:
            raise ValueError('The method needs to be one of: {}'
                             .format(implemented_methods))

        if iterations is None:
            guess = t / self.dt
            if method == 'previous':
                iteration = np.floor(guess)
            if method == 'closest':
                iteration = np.round(guess)
            if method == 'next':
                iteration = np.ceil(guess)
            return np.uint64(iteration)
        else:
            if type(iterations) is not np.ndarray:
                raise ValueError('iterations must to be a numpy array!')

        iterations_sorted = np.sort(iterations)
        times_sorted = iterations_sorted * self.dt

        next_i = np.argmax(times_sorted > t)
        if next_i > 0:
            prev_i = next_i - 1
        else:
            prev_i = 0

        if method == 'previous':
            iteration = iterations_sorted[prev_i]
            if times_sorted[prev_i] > t:
                raise IndexError('Time t not found in valid range!')
        if method == 'closest':
            closest_i = np.abs(times_sorted - t).argmin()
            iteration = iterations_sorted[closest_i]
        if method == 'next':
            iteration = iterations_sorted[next_i]
            if times_sorted[next_i] < t:
                raise IndexError('Time t not found in valid range!')

        return np.uint64(iteration), iteration * self.dt
