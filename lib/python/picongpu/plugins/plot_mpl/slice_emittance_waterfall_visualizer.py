"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sophie Rudat
License: GPLv3+
"""

from picongpu.plugins.emittance import Emittance
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as\
    BaseVisualizer, plt
import numpy as np
from matplotlib.colors import LogNorm


class Visualizer(BaseVisualizer):
    """
    Class for creation of waterfall plot with the slice emittance value 
    for each y_slice (x-axis) and iteration (y-axis).
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
        self.cbar = None

    def _create_data_reader(self, run_directory):
        """
        Implementation of base class function.
        """
        return Emittance(run_directory)

    def _create_plt_obj(self, ax):
        """
        Implementation of base class function.
        Turns 'self.plt_obj' into a matplotlib.pyplot.plot object.
        """
        slice_emit, y_slices, iteration = self.data
        np_data = np.zeros((len(y_slices), len(iteration)))
        for index, ts in enumerate(iteration):
            np_data[:, index] = slice_emit[ts][1:]
        max_iter = max(iteration* 1.39e-16 * 1.e12)
        self.plt_obj = ax.imshow(np_data.T*1.e6,aspect="auto", 
                                norm=LogNorm(), origin="lower", 
                                vmin=1e-1, vmax=1e2,
                                extent=(0,max(y_slices*1.e6),0,max_iter))
        self.plt_lin = ax.axhline(self.itera* 1.39e-16 * 1.e12,
                                  color='#FF6600')
        self.cbar = plt.colorbar(self.plt_obj, ax=ax)
        self.cbar.set_label(r'emittance [pi mm mrad]')
        ax.set_xlabel('y-slice [µm]')
        ax.set_ylabel('time [ps]')

    def _update_plt_obj(self):
        """
        Implementation of base class function.
        """
        slice_emit, y_slices, iteration = self.data
        np_data = np.zeros((len(y_slices), len(iteration)))
        for index, ts in enumerate(iteration):
            np_data[:, index] = slice_emit[ts][1:]
        self.plt_obj.set_data(np_data.T*1.e6)
        self.plt_lin.remove()
        self.plt_lin=self.ax.axhline(self.itera* 1.39e-16 * 1.e12,color='#FF6600')
        self.cbar.update_normal(self.plt_obj)


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
        self.ax = self._ax_or_gca(ax)
        self.itera = kwargs.get('iteration')
        kwargs['iteration']=None
        super(Visualizer, self).visualize(ax, **kwargs)
        species = kwargs.get('species')
        species_filter = kwargs.get('species_filter', 'all')
        ax.set_title('slice emittance for species ' +
                     species + ', filter = ' + species_filter)

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
