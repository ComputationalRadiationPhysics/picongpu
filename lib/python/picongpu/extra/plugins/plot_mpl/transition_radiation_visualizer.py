"""
This file is part of PIConGPU and based on energy_histogram_visualizer.py.

Authors: Finn-Ole Carstens, Sebastian Starke
"""

from picongpu.plugins.data import TransitionRadiationData
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as \
    BaseVisualizer
from matplotlib.ticker import FixedLocator
import numpy as np
import scipy.constants as const


class Visualizer(BaseVisualizer):
    """
    Class for creation of different plots for analysis of the CTR plugin in
    PIConGPU.
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
        super().__init__(TransitionRadiationData, run_directories, ax)
        self.type = None

    def _create_plt_obj(self, idx):
        """
           Implementation of base class function.
           Turns 'self.plt_obj' into a matplotlib.pyplot.plot object.
           """
        if self.type == "spectrum":
            omegas, spectral_power = self.data[idx]
            self.plt_obj[idx] = self.ax.plot(omegas, spectral_power)
        elif self.type == "sliceovertheta":
            thetas, spectral_power = self.data[idx]
            self.plt_obj[idx] = self.ax.plot(thetas, spectral_power)
        elif self.type == "sliceoverphi":
            phis, spectral_power = self.data[idx]
            self.plt_obj[idx] = self.ax.plot(phis, spectral_power)
        elif self.type == "heatmap":
            theta_mesh, phi_mesh, spectral_power = self.data[idx]
            im = self.plt_obj[idx] = self.ax.pcolormesh(theta_mesh, phi_mesh,
                                                        spectral_power)
            self.ax.get_figure().colorbar(im,
                                          label=r"Spectral Power $d^2 W / "
                                                r"d\omega d\Omega$ [Js]")

    def _update_plt_obj(self, idx):
        """
        Implementation of base class function.
        """
        # @TODO needed for jupyter widget
        return 0

    def visualize(self, **kwargs):
        """
        Creates a plot on the provided axes object for
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
            type: string
                name of figure type. valid figure types are:
                    'spectrum' - (default) plots transition radiation spectrum
                        at angles theta and phi over the frequency omega
                    'sliceovertheta' - shows angular distribution of
                    transition radiation
                        at a fixed angle phi and frequency omega
                    'sliceoverphi' - shows angular distribution of
                    transition radiation
                        at a fixed angle theta and frequency omega
                    'heatmap' - shows angular distribution as heatmap over
                    both observation angles
            phi: int
                index of polar angle for a fixed value
            theta: int
                index of azimuth angle for a fixed value
            omega: int
                index of frequency for a fixed value, pointless in a spectrum
        """
        self.type = kwargs["type"]
        super().visualize(**kwargs)

    def adjust_plot(self, **kwargs):
        """
        Implementation of base class function.
        """
        species = kwargs["species"]
        if self.type == "spectrum":
            self.ax.set_xlabel(r"Frequency $\omega$ [1/s]")
            self.ax.set_ylabel(
                r"Spectral Power $d^2 W / d\omega d\Omega$ [Js]")
            self.ax.set_xscale("log")
            self.ax.set_yscale("log")
            self.ax.set_title("Transition Radiation Spectrum for " + species)

            # care for beautiful figure top axes
            # activate top axes as a log axes as clone from bottom axes
            axtop = self.ax.twiny()
            axtop.set_xscale("log")
            axtop.autoscale(False)
            axtop.set_xlim(self.ax.get_xlim())

            # necessary functions for calculation from lambda to omega and
            # vice versa
            def calc_lambda(_omega):
                return 2 * np.pi * const.c / _omega

            def calc_omega(_lambda):
                return 2 * np.pi * const.c / _lambda

            # calculate limits for top ticks
            lambda_min = np.ceil(np.log10(calc_lambda(self.ax.get_xlim()[1])))
            lambda_max = np.floor(np.log10(calc_lambda(self.ax.get_xlim()[0])))
            # calculate tick locations
            lambda_locations = np.logspace(lambda_min, lambda_max,
                                           np.abs(lambda_max - lambda_min) + 1)
            # set tick locations of top axes
            axtop.set_xticks(calc_omega(lambda_locations))

            # calculate positions for minor and major ticks for the beauty
            # of the plot
            nmaj = 200
            nmin = 10
            minorlocs = np.linspace(0.1, 1, nmin, endpoint=True)
            majorlocs = np.linspace(99, -100, nmaj, endpoint=True)
            alllocs = np.zeros(nmaj * nmin)
            for i in range(nmaj):
                for j in range(nmin):
                    alllocs[i * nmin + j] = minorlocs[j] * 10 ** majorlocs[i]

            alllocs = calc_omega(alllocs)

            axtop.xaxis.set_minor_locator(FixedLocator(alllocs))

            # necessary function for labelling of top axes
            def maketentothepower(x):
                if x == 0:
                    return "$0$"
                exponent = np.int32(np.log10(x))
                return r"$10^{{ {:2d} }}$".format(exponent)

            # create names for top ticks with log scale
            lambda_names = []
            for i in range(len(lambda_locations)):
                lambda_names.append(maketentothepower(lambda_locations[i]))

            # set tick labels and ax label for top label
            axtop.set_xticklabels(lambda_names)
            axtop.set_xlabel(r"Wavelength $\lambda [m]$")
        elif self.type == "sliceovertheta":
            self.ax.set_title(
                "Angular transition radiation distribution for " + species)
            self.ax.set_xlabel(r"Detector Angle $\theta$")
            self.ax.set_ylabel(r"Spectral Power "
                               r"$d^2 W / d\omega d\Omega$ [Js]")
        elif self.type == "sliceoverphi":
            self.ax.set_title(
                "Angular transition radiation distribution for " + species)
            self.ax.set_xlabel(r"Detector Angle $\phi$")
            self.ax.set_ylabel(r"Spectral Power "
                               r"$d^2 W / d\omega d\Omega$ [Js]")
        elif self.type == "heatmap":
            self.ax.set_title(
                "Angular transition radiation distribution for " + species)
            self.ax.set_xlabel(r"Detector Angle $\theta$")
            self.ax.set_ylabel(r"Detector Angle $\phi$")


if __name__ == "__main__":
    def main():
        import sys
        import getopt
        import matplotlib.pyplot as plt

        def usage():
            print("usage:")
            print("python", sys.argv[0], "\n",
                  "\t-p, --path\t<path to run_directory>\n"
                  "\t-i, --iteration\t<iteration>\n"
                  "\t-s, --species\t<particle species>\n"
                  "\t-t, --type\t<figure type>\n"
                  "\t\tavailable figure types:\n"
                  "\t\t -'spectrum'\n"
                  "\t\t -'sliceovertheta'\n"
                  "\t\t -'sliceoverphi'\n"
                  "\t\t -'heatmap'\n"
                  "\t-P, --phi\t<index of polar angle phi>\n"
                  "\t-T, --theta\t<index of azimuth angle theta>\n"
                  "\t-O, --omega\t<index of frequency omega>")

        path = None
        iteration = None
        species = None
        type = None

        # Values for a plot with a fixed angle or frequency
        phi = None
        theta = None
        omega = None

        try:
            opts, args = getopt.getopt(sys.argv[1:], "hp:i:s:t:P:T:O:",
                                       ["help", "path", "iteration",
                                        "species", "type", "phi",
                                        "theta", "omega"])
        except getopt.GetoptError as err:
            print(err)
            usage()
            sys.exit(2)

        # go through kwargs
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
            elif opt in ["-t", "--type"]:
                type = arg
            elif opt in ["-P", "--phi"]:
                phi = arg
            elif opt in ["-T", "--theta"]:
                theta = arg
            elif opt in ["-O", "--omega"]:
                omega = arg

        # check that we got all args that we need
        if path is None or iteration is None:
            print("Path to 'run' directory and iteration have to be provided!")
            usage()
            sys.exit(2)
        if species is None:
            species = "e"
            print("Particle species was not given, will use", species)
        if type is None:
            type = "spectrum"
            print("Figure type was not given, will use", type)

        # create pyplot axes object and visualize data
        _, ax = plt.subplots(1, 1)
        Visualizer(path, ax).visualize(iteration=iteration, species=species,
                                       type=type,
                                       phi=phi, theta=theta, omega=omega)
        # plot layout magic for the beauty of the plot
        plt.tight_layout()
        plt.show()

    main()
