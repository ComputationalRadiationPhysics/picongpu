"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

import matplotlib.pyplot as plt


class Visualizer(object):
    """
    Abstract base class for matplotlib visualizers that implements
    the visualization logic.
    Classes that derive from this class need to write their own implementations
    for the following functions in order to work:
        _create_data_reader(self, run_directory)
        _create_plt_obj(self, ax)
        _update_plt_obj(self)
    """

    def __init__(self, run_directory):
        """
        Initialize the reader and data as member parameters.

        Parameters
        ----------
        run_directory : string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        """
        if run_directory is None:
            raise ValueError('The run_directory parameter can not be None!')

        self.plt_obj = None
        self.data_reader = self._create_data_reader(run_directory)
        self.data = None

    def _create_data_reader(self, run_directory):
        """
        Needs to return an instance of a picongpu data reader
        (as defined in the ../plugin directory) which implements
        a 'get()' method.
        """
        raise NotImplementedError

    def _create_plt_obj(self, ax):
        """
        Sets 'self.plt_obj' to an instance of a matplotlib.artist.Artist
        object (or derived classes) which can be updated
        by feeding new data into it.
        Only called on the first call for visualization.

        Parameters
        ----------
        ax: matplotlib.axes.Axes instance
        """
        raise NotImplementedError

    def _update_plt_obj(self):
        """
        Take the 'self.data' member, interpret it and feed it into the
        'self.plt_obj'.
        """
        raise NotImplementedError

    def _ax_or_gca(self, ax):
        """
        Returns the passed ax if it is not None or the current
        matplotlib axes object otherwise.

        Parameters
        ----------
        ax: None or matplotlib.axes.Axes instance
        Returns
        -------
        A matplotlib.axes.Axes instance
        """

        return ax or plt.gca()

    def visualize(self, ax=None, **kwargs):
        """
        1. Creates the 'plt_obj' if it does not exist
        2. Fills the 'data' parameter by using the reader
        3. Updates the 'plt_obj' with the new data.
        """
        if ax is None:
            raise ValueError("A matplotlib axes object needs to be passed!")

        self.data = self.data_reader.get(**kwargs)
        if self.plt_obj is None:
            self._create_plt_obj(ax)
        else:
            self._update_plt_obj()
