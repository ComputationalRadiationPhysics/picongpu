"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""


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
        # need to be set in derived classes
        self.data_file_prefix = None
        self.data_file_suffix = None

    def get_data_path(self, **kwargs):
        """
        Returns
        -------
        A string with the path to the underlying data file.
        """
        raise NotImplementedError

    def get_iterations(self, **kwargs):
        """
        Returns
        -------
        An array with unsigned integers of iterations for which
        data is available.
        """
        raise NotImplementedError

    def get(self, **kwargs):
        """
        Returns
        -------
        The data for the requested parameters in a plugin
        dependent format and type.
        """
        raise NotImplementedError
