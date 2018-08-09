"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

from matplotlib.colors import LogNorm
import numpy as np

from picongpu.plugins.phase_space import PhaseSpace
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as\
    BaseVisualizer, plt


class Visualizer(BaseVisualizer):
    """
    Class for creating a matplotlib plot of phase space diagrams.
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

        # for unit-conversion from SI (taken from picongpu readthedocs)
        self.mu = 1.e6
        self.e_mc_r = 1. / (9.1e-31 * 2.9979e8)
        self.cbar = None

    def _create_data_reader(self, run_directory):
        """
        Implementation of base class function.
        """
        return PhaseSpace(run_directory)

    def _create_plt_obj(self, ax):
        """
        Implementation of base class function.
        Turns 'self.plt_obj' into a matplotlib.image.AxesImage object.
        """
        dat, meta = self.data
        self.plt_obj = ax.imshow(
            np.abs(dat).T * meta.dV,
            extent=meta.extent * [self.mu, self.mu, self.e_mc_r, self.e_mc_r],
            interpolation='nearest',
            aspect='auto',
            origin='lower',
            norm=LogNorm()
        )
        self.cbar = plt.colorbar(self.plt_obj, ax=ax)
        self.cbar.set_label(
                r'$Q / \mathrm{d}r \mathrm{d}p$ [$\mathrm{C s kg^{-1} m^{-2}}$] ')
        ax.set_xlabel(r'${0}$ [${1}$]'.format(meta.r, "\mathrm{\mu m}"))
        ax.set_ylabel(r'$p_{0}$ [$\beta\gamma$]'.format(meta.p))

    def _update_plt_obj(self):
        """
        Implementation of base class function.
        """
        dat, meta = self.data
        self.plt_obj.set_data(np.abs(dat).T * meta.dV)
        self.plt_obj.autoscale()
        self.cbar.update_normal(self.plt_obj)

    def visualize(self, ax=None, **kwargs):
        """
        Creates a phase space plot on the provided axes object for
        the data of the given iteration using matpotlib.

        Parameters
        ----------
        ax: matplotlib axes object
            the part of the figure where this plot will be shown.
        kwargs: dict with possible additional keyword args. Valid are:
            iteration: int
                the iteration number for which data will be plotted.
            species : string
                short name of the particle species, e.g. 'e' for electrons
                (defined in ``speciesDefinition.param``)
            species_filter: string
                name of the particle species filter, default is 'all'
                (defined in ``particleFilters.param``)
            ps : string
                phase space selection in order: spatial, momentum component,
                e.g. 'ypy' or 'ypx'
        """
        ax = self._ax_or_gca(ax)
        super(Visualizer, self).visualize(ax, **kwargs)

    def clear_cbar(self):
        """Clear colorbar if present."""
        if self.cbar is not None:
            self.cbar.remove()


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
                " -m <phase space selection>")

        path = None
        iteration = None
        species = None
        filtr = None
        momentum = None
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hp:i:s:f:m:", [
                "help", "path", "iteration", "species", "filter", "momentum"])
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
        if filtr is None:
            filtr = 'all'
            print("Species filter was not given, will use", filtr)
        if momentum is None:
            momentum = 'ypy'
            print("Momentum term was not given, will use", momentum)

        fig, ax = plt.subplots(1, 1)
        Visualizer(path).visualize(ax, iteration=iteration, species=species,
                                   species_filter=filtr, ps=momentum)
        plt.show()

    main()
