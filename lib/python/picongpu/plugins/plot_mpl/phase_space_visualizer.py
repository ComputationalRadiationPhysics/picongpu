"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

from matplotlib.colors import LogNorm
import numpy as np

from picongpu.plugins.data import PhaseSpaceData
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as\
    BaseVisualizer

import matplotlib.pyplot as plt


class Visualizer(BaseVisualizer):
    """
    Class for creating a matplotlib plot of phase space diagrams.
    """

    def __init__(self, run_directory=None, ax=None):
        """
        Parameters
        ----------
        run_directory : string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``).
            If None, the user is responsible for providing run_directories
            later on via set_run_directories() before calling visualize().
        ax: matplotlib.axes
        """
        # TODO: remove this and get it from the readers metadata
        # since it seems only correct for species='e' anyway.
        self.mu = 1.e6
        self.e_mc_r = 1. / (9.1e-31 * 2.9979e8)
        self.cbar = None

        super().__init__(PhaseSpaceData, run_directory, ax)

    def _check_and_fix_run_dirs(self, run_directories):
        """
        Overridden from base class. Makes sure to only accept
        a single simulation's run_directory.
        """
        base_checked = super()._check_and_fix_run_dirs(run_directories)

        # Fail if more than one run_directory since plotting
        # several imshow plots at the same time does not make sense!
        if len(base_checked) > 1:
            raise ValueError("This visualizer only supports plotting a single"
                             " simulation! Parameter 'run_directory' can"
                             " contain only a single element!")

        return base_checked

    def _create_plt_obj(self, idx):
        """
        Implementation of base class function.
        Turns 'self.plt_obj' into a matplotlib.image.AxesImage object.
        """
        dat, meta = self.data[idx]
        self.plt_obj[idx] = self.ax.imshow(
            np.abs(dat).T * meta.dV,
            extent=meta.extent * [self.mu, self.mu, self.e_mc_r, self.e_mc_r],
            interpolation='nearest',
            aspect='auto',
            origin='lower',
            norm=LogNorm())
        self.cbar = plt.colorbar(self.plt_obj[idx], ax=self.ax)
        self.cbar.set_label(
            r'$Q / \mathrm{d}r \mathrm{d}p$ [$\mathrm{C s kg^{-1} m^{-2}}$] ')
        self.ax.set_xlabel(r'${0}$ [${1}$]'.format(meta.r, r'\mathrm{\mu m}'))
        self.ax.set_ylabel(r'$p_{0}$ [$\beta\gamma$]'.format(meta.p))

    def _update_plt_obj(self, idx):
        """
        Implementation of base class function.
        """
        dat, meta = self.data[idx]
        self.plt_obj[idx].set_data(np.abs(dat).T * meta.dV)
        self.plt_obj[idx].autoscale()
        self.cbar.update_normal(self.plt_obj[idx])

    def visualize(self, **kwargs):
        """
        Creates a phase space plot on the provided axes object for
        the data of the given iteration using matpotlib.

        Parameters
        ----------
        kwargs: dict with possible additional keyword args. Valid are:
            iteration: int
                the iteration number for which data will be plotted.
            time: float
                simulation time.
                Only one of 'iteration' or 'time' should be passed!
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
        super().visualize(**kwargs)

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

        _, ax = plt.subplots(1, 1)
        Visualizer(path, ax).visualize(iteration=iteration, species=species,
                                       species_filter=filtr, ps=momentum)
        plt.show()

    main()
