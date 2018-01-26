from picongpu.plugins.phase_space import PhaseSpace

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


class Visualizer(object):
    """
    Class for creating a matplotlib plot of phase space diagrams.
    """

    def __init__(self, layout_config):
        """
        Parameters
        ----------
        layout_config: dictionary
            containing information about particle species (e.g. 'e' for
            electrons) and phase space momentum string (e.g. xpx, ypy, ypx)
        """

        self.species = layout_config.get('particle_species', 'e')
        self.ps = layout_config.get('phase_space_momentum', 'ypy')

        # for unit-conversion from SI (taken from picongpu readthedocs)
        self.mu = 1.e6
        self.e_mc_r = 1. / (9.1e-31 * 2.9979e8)

        self.phase_space = None
        self.plt_obj = None

    def visualize(self, path, iteration, ax):
        """
        Creates a phase space plot on the provided axes object for
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

        self.phase_space = PhaseSpace(path)

        data, meta = self.phase_space.get(
            species=self.species,
            ps=self.ps,
            iteration=iteration)

        if self.plt_obj is None:
            self.plt_obj = ax.imshow(
                np.abs(data).T * meta.dV,
                extent=meta.extent *
                [self.mu, self.mu, self.e_mc_r, self.e_mc_r],
                interpolation='nearest',
                aspect='auto',
                origin='lower',
                norm=LogNorm()
            )
        else:
            self.plt_obj.set_data(np.abs(data).T * meta.dV)

        # prevent multiple rendering of colorbar
        if not self.plt_obj.colorbar:
            cbar = plt.colorbar(self.plt_obj, ax=ax)
            cbar.set_label(
                r'$Q / \mathrm{d}r \mathrm{d}p$ [$\mathrm{C s kg^{-1} m^{-2}}$]')

        ax.set_xlabel(r'${0}$ [${1}$]'.format(meta.r, "\mathrm{\mu m}"))
        ax.set_ylabel(r'$p_{0}$ [$\beta\gamma$]'.format(meta.p))


if __name__ == '__main__':
    import sys

    def usage():
        print("usage:")
        print(
            "python", sys.argv[0], "-p <path to run directory> -i <iteration>\
            -s <particle species> -m <momentum term>")

    def main():

        import getopt

        path = None
        iteration = None
        species = None
        momentum = None

        try:
            opts, args = getopt.getopt(sys.argv[1:], "hp:i:s:m:", [
                "help", "path", "iteration", "species", "momentum"])
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
            elif opt in ["-m", "--momentum"]:
                momentum = arg

        # check that we got all args that we need
        if path is None or iteration is None:
            print("Path to 'run' directory and iteration have to be provided!")
            usage()
            sys.exit(2)

        if species is None:
            species = 'e'
            print("Particle species was not given, will use", species)

        if momentum is None:
            momentum = 'ypy'
            print("Momentum term was not given, will use", momentum)

        layout_config = {
            'particle_species': species,
            'phase_space_momentum': momentum
        }

        fig, ax = plt.subplots(1, 1)
        Visualizer(layout_config).visualize(path, iteration, ax)
        plt.show()

    main()
