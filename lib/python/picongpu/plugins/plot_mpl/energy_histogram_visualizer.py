"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""
import numpy as np

from picongpu.plugins.data import EnergyHistogramData
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as\
    BaseVisualizer
from warnings import warn


class Visualizer(BaseVisualizer):
    """
    Class for creation of histogram plots on a logscaled y-axis.
    """

    def __init__(self, run_directories=None, ax=None):
        """
        Parameters
        ----------
        run_directories: list of tuples of length 2
            or single tuple of length 2.
            Each tuple is of the following form (sim_name, sim_path)
            and consists of strings.
            sim_name is a short string used e.g. in plot legends.
            sim_path leads to the run directory of PIConGPU
            (the path before ``simOutput/``).
            If None, the user is responsible for providing run_directories
            later on via set_run_directories() before calling visualize().
        ax: matplotlib.axes
        """
        super().__init__(EnergyHistogramData, run_directories, ax)

    def _create_plt_obj(self, idx):
        """
        Implementation of base class function.
        Turns 'self.plt_obj' into a matplotlib.pyplot.plot object.
        """
        # NOTE: for ax.semilogy one can also provide matrices Bins, Counts
        # where columns are the separate data to be plotted.
        # Returned would then be a list of plt_objects

        counts, bins, iteration, dt = self.data[idx]
        label = self.sim_labels[idx]

        if np.all(counts == 0.):
            warn("All counts were 0 for {}. ".format(label) +
                 "No log-plot can be created!")
            return

        self.plt_obj[idx] = self.ax.semilogy(
            bins, counts, nonposy='clip', label=label,
            color=self.colors[idx])[0]

    def _update_plt_obj(self, idx):
        """
        Implementation of base class function.
        """
        counts, bins, iteration, dt = self.data[idx]
        label = self.sim_labels[idx]

        if np.all(counts == 0.):
            warn("All counts were 0 for {}. ".format(label) +
                 "Log-plot will not be updated!")
            return

        self.plt_obj[idx].set_data(bins, counts)

    def visualize(self, **kwargs):
        """
        Creates a semilogy plot on the provided axes object for
        the data of the given iteration using matpotlib.

        Parameters
        ----------
        kwargs: dictionary with further keyword arguments, valid are:
            species: string
                short name of the particle species, e.g. 'e' for electrons
                (defined in ``speciesDefinition.param``)
            iteration: int
                number of the iteration
            time: float
                simulation time.
                Only one of 'iteration' or 'time' should be passed!
            species_filter: string
                name of the particle species filter, default is 'all'
                (defined in ``particleFilters.param``)

        """
        super().visualize(**kwargs)

    def adjust_plot(self, **kwargs):
        species = kwargs['species']
        species_filter = kwargs.get('species_filter', 'all')

        self._legend()
        self.ax.relim()
        self.ax.autoscale_view(True, True, True)
        self.ax.set_xlabel('Energy [keV]')
        self.ax.set_ylabel('Counts')
        self.ax.set_title('Energy Histogram for species ' +
                          species + ', filter = ' + species_filter)

    def _legend(self):
        # draw the legend only for those lines for which there is data.
        # colors will not change in between simulations since they are
        # tied to the data readers index directly.
        handles = []
        labels = []
        for plt_obj, lab in zip(self.plt_obj, self.sim_labels):
            if plt_obj is not None:
                handles.append(plt_obj)
                labels.append(lab)

        self.ax.legend(handles, labels)


if __name__ == '__main__':

    def main():
        import sys
        import getopt
        import matplotlib.pyplot as plt

        def usage():
            print("usage:")
            print(
                "python", sys.argv[0],
                "-p <path to run_directory> -i <iteration>"
                " -s <particle species> -f <species_filter>")

        path = None
        iteration = None
        species = None
        filtr = None

        try:
            opts, args = getopt.getopt(sys.argv[1:], "hp:i:s:f:", [
                "help", "path", "iteration", "species", "filter"])
        except getopt.GetoptError as err:
            print(err)
            usage()
            sys.exit(2)

        for opt, arg in opts:
            if opt in ["-h", "--help"]:
                usage()
                sys.exit()
            elif opt in ["-p", "--path"]:
                path = arg
            elif opt in ["-i", "--iteration"]:
                iteration = int(arg)
            elif opt in ["-s", "--species"]:
                species = arg
            elif opt in ["-f", "--filter"]:
                filtr = arg

        # check that we got all args that we need
        if path is None or iteration is None:
            print("Path to 'run' directory and iteration have to be provided!")
            usage()
            sys.exit(2)
        if species is None:
            species = 'e'
            print("Particle species was not given, will use", species)
        if filtr is None:
            filtr = 'all'
            print("Species filter was not given, will use", filtr)

        _, ax = plt.subplots(1, 1)
        Visualizer(path, ax).visualize(iteration=iteration, species=species,
                                       species_filter=filtr)
        plt.show()

    main()
