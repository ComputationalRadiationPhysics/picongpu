from picongpu.plugins.energy_histogram import EnergyHistogram
import matplotlib.pyplot as plt


class Visualizer:
    """
    Class for creation of histogram plots on a logscaled y-axis.
    """

    def __init__(self, layout_config):
        """
        Parameters
        ----------
        Parameters
        ----------
        layout_config: dictionary
            containing information about particle species (e.g. 'e' for
            electrons)
        """

        self.species = layout_config.get('particle_species', 'e')
        self.energy_histogram = None
        self.plt_obj = None

    def visualize(self, path, iteration, ax):
        """
        Creates a semilogy plot on the provided axes object for
        the data of the given iteration using matpotlib.

        Parameters
        ----------
        path: string
            full path to the 'run' subdirectory of an experiment.
        iteration: int
            the iteration number for which data will be plotted.

        ax: matplotlib axes object
            the part of the figure where this plot will be shown.
        """

        self.energy_histogram = EnergyHistogram(path)

        # catch exception when histogram file does not exist yet
        try:
            self.energy_histogram.get_data_path(species=self.species)
        except IOError:
            print(
                "No histogram file exists yet. Wait and hit visualize again after a while!")
            return

        counts, energy_bins = self.energy_histogram.get(
            species=self.species, iteration=iteration)
        # plot energy per bin after cleaning the plot

        # if it is the first time for plotting then object is
        # created, otherwise only data is updated
        if self.plt_obj is None:
            self.plt_obj = ax.semilogy(energy_bins, counts, nonposy='clip')[0]
        else:
            self.plt_obj.set_data(energy_bins, counts)

        ax.set_xlabel('Energy')
        ax.set_ylabel('Count')
        ax.set_title('Energy Histogram for species ' +
                     self.species + ', iteration ' + str(iteration))


if __name__ == '__main__':
    import sys

    def usage():
        print("usage:")
        print(
            "python", sys.argv[0], "-p <path to run directory> -i <iteration> -s <particle species>")

    def main():

        import getopt

        path = None
        iteration = None
        species = None

        try:
            opts, args = getopt.getopt(sys.argv[1:], "hp:i:s:", [
                "help", "path", "iteration", "species"])
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

        # check that we got all args that we need
        if path is None or iteration is None:
            print("Path to 'run' directory and iteration have to be provided!")
            usage()
            sys.exit(2)
        if species is None:
            species = 'e'
            print("Particle species was not given, will use", species)

        layout_config = {
            'particle_species': species
        }

        fig, ax = plt.subplots(1, 1)
        Visualizer(layout_config).visualize(path, iteration, ax)
        plt.show()

    main()
