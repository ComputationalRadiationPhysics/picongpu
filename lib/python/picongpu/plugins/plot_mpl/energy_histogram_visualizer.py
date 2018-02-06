"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

from picongpu.plugins.energy_histogram import EnergyHistogram
import matplotlib.pyplot as plt


class Visualizer:
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
        if run_directory is None:
            raise ValueError('The run_directory parameter can not be None!')

        self.energy_histogram = EnergyHistogram(run_directory)
        self.plt_obj = None

    def visualize(self, iteration, ax, species='e', species_filter="all",
                  **kwargs):
        """
        Creates a semilogy plot on the provided axes object for
        the data of the given iteration using matpotlib.

        Parameters
        ----------
        iteration: int
            the iteration number for which data will be plotted.
        ax: matplotlib axes object
            the part of the figure where this plot will be shown.
        species : string
            short name of the particle species, e.g. 'e' for electrons
            (defined in ``speciesDefinition.param``)
        species_filter: string
            name of the particle species filter, default is 'all'
            (defined in ``particleFilters.param``)
        kwargs: dict
            possible additional keyword args to pass to the reader.
        """

        counts, energy_bins = self.energy_histogram.get(
            species=species,
            species_filter=species_filter,
            iteration=iteration,
            **kwargs)

        # if it is the first time for plotting then object is
        # created, otherwise only data is updated
        if self.plt_obj is None:
            self.plt_obj = ax.semilogy(energy_bins, counts, nonposy='clip')[0]
        else:
            self.plt_obj.set_data(energy_bins, counts)

        ax.set_xlabel('Energy [keV]')
        ax.set_ylabel('Count')
        ax.set_title('Energy Histogram for species ' +
                     species + ', filter = ' + species_filter +
                     ', iteration ' + str(iteration))


if __name__ == '__main__':

    def main():
        import sys
        import getopt

        def usage():
            print("usage:")
            print(
                "python", sys.argv[0],
                "-p <path to run_directory> -i <iteration>"
                " -s <particle species> -f <species_filter>"
                " -o <include_overflow>")

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
        Visualizer(path).visualize(iteration, ax, species=species,
                                   species_filter=filtr)
        plt.show()

    main()
