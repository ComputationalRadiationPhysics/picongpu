"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

from picongpu.plugins.energy_histogram import EnergyHistogram
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as\
    BaseVisualizer, plt


class Visualizer(BaseVisualizer):
    """
    Class for creation of histogram plots on a logscaled y-axis.
    """

    def __init__(self, run_directory):
        """
        Parameters
        ----------
        run_directory : string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        """
        super(Visualizer, self).__init__(run_directory)

    def _create_data_reader(self, run_directory):
        """
        Implementation of base class function.
        """
        return EnergyHistogram(run_directory)

    def _create_plt_obj(self, ax):
        """
        Implementation of base class function.
        Turns 'self.plt_obj' into a matplotlib.pyplot.plot object.
        """
        counts, bins = self.data
        self.plt_obj = ax.semilogy(bins, counts, nonposy='clip')[0]

    def _update_plt_obj(self):
        """
        Implementation of base class function.
        """
        counts, bins = self.data
        self.plt_obj.set_data(bins, counts)
        ax = self._ax_or_gca(None)
        ax.relim()
        ax.autoscale_view(True, True, True)

    def visualize(self, ax=None, **kwargs):
        """
        Creates a semilogy plot on the provided axes object for
        the data of the given iteration using matpotlib.

        Parameters
        ----------
        iteration: int
            the iteration number for which data will be plotted.
        ax: matplotlib axes object
            the part of the figure where this plot will be shown.
        kwargs: dictionary with further keyword arguments, valid are:
            species: string
                short name of the particle species, e.g. 'e' for electrons
                (defined in ``speciesDefinition.param``)
            iteration: int
                number of the iteration
            species_filter: string
                name of the particle species filter, default is 'all'
                (defined in ``particleFilters.param``)

        """
        ax = self._ax_or_gca(ax)
        # this already throws error if no species or iteration in kwargs
        super(Visualizer, self).visualize(ax, **kwargs)
        iteration = kwargs.get('iteration')
        species = kwargs.get('species')
        species_filter = kwargs.get('species_filter', 'all')
        if iteration is None or species is None:
            raise ValueError("Iteration and species have to be provided as\
            keyword arguments!")

        ax.set_xlabel('Energy [keV]')
        ax.set_ylabel('Counts')
        ax.set_title('Energy Histogram for species ' +
                     species + ', filter = ' + species_filter)


if __name__ == '__main__':

    def main():
        import sys
        import getopt

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

        fig, ax = plt.subplots(1, 1)
        Visualizer(path).visualize(ax, iteration=iteration, species=species,
                                   species_filter=filtr)
        plt.show()

    main()
