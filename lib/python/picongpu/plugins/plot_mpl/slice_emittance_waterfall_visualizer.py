"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Sophie Rudat, Sebastian Starke
License: GPLv3+
"""

from picongpu.plugins.data import EmittanceData
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as\
    BaseVisualizer
from mpl_toolkits.axes_grid1 import make_axes_locatable
from picongpu.plugins.plot_mpl.utils import get_different_colormaps
import numpy as np
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt


class Visualizer(BaseVisualizer):
    """
    Class for creation of waterfall plot with the slice emittance value
    for each y_slice (x-axis) and iteration (y-axis).
    """

    def __init__(self, run_directories=None, ax=None):
        """
        Parameters
        ----------
        run_directory : list of tuples of length 2
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
        self.colorbars = None
        # the separate colorbar axes
        self.colorbar_axes = None
        super().__init__(EmittanceData, run_directories, ax)
        self.plt_lin = None  # plot line at current itteration
        self.cur_iteration = None

    def _init_members(self, run_directories):
        """
        Overridden from base class. Need to create colorbar instances
        """
        super()._init_members(run_directories)
        self._init_colorbars(run_directories)
        self._init_colorbar_axes(run_directories)

    def _clean_ax(self):
        # we have to remove the colorbars from our own plt_objs
        if self.plt_obj is not None:
            for idx in range(len(self.plt_obj)):
                # give back the space to the figure which was
                # previously used by the colorbars
                if self.plt_obj[idx] is not None:
                    self.plt_obj[idx].colorbar.remove()
                # NOTE: now the self.plt_obj[idx] has an invalid
                # reference to its colorbar ax so if we tried to do
                # self.ax.images[idx].colorbar.remove() for one of our
                # own images (i.e. images[idx] == self.plt_obj[some_idx])
                # mpl would fail since it calls cbar.ax.figure.delaxes.
                # This is why we can't call the base class function!

                # NOTE: invalidation of colorbars and axes will be done
                # in the _init functions called after this one

        # clear potentially occuring colorbars that are not from this object
        # i.e. the ax was used by a different object before
        for plt_obj in self.ax.images:
            if self.plt_obj is not None:
                if plt_obj not in self.plt_obj:
                    # there is an image that comes from other visualizer
                    if plt_obj.colorbar is not None:
                        plt_obj.colorbar.remove()
            else:
                # we can delete since it is not from our object
                if plt_obj.colorbar is not None:
                    plt_obj.colorbar.remove()

        # this removes all imshow images or previous plots
        # regardless if they were our own or from some other object
        self.ax.clear()

    def _init_colorbars(self, run_directories):
        self.colorbars = [None] * len(run_directories)

    def _init_colorbar_axes(self, run_directories):
        self.colorbar_axes = [None] * len(run_directories)

        divider = make_axes_locatable(self.ax)
        for i in range(len(self.colorbar_axes)):
            cax = divider.append_axes(
                "right", size="5%", pad=0.5)
            self.colorbar_axes[i] = cax

    def _init_colors(self, run_directories):
        """
        Overridden from base class. Create colormaps instead
        of colors since this is more useful for imshow plots.
        """
        self.colors = get_different_colormaps(len(run_directories))

    def _remove_colorbar(self, idx):
        """
        Remove the colorbar for plot obj at idx.
        """
        # do not call self.colorbars[idx].remove() since that removes the
        # ax of the colorbar from the figure which we don't want
        self.colorbars[idx].ax.clear()
        # deactivate the axis labels here so we don't have them as leftovers
        self.colorbars[idx].ax.axis("off")
        self.colorbars[idx] = None

    def _remove_plt_obj(self, idx):
        """
        Overridden from base class.
        Remove the colorbars before removing the plot object.
        This order is necessary since otherwise matplotlib complains.
        """
        self._remove_colorbar(idx)
        # clear the plt_obj and set it to None
        super()._remove_plt_obj(idx)

    def _create_plt_obj(self, idx):
        """
        Implementation of base class function.
        Turns 'self.plt_obj' into a matplotlib.pyplot.plot object.
        """
        slice_emit, y_slices, all_iterations, dt = self.data[idx]
        np_data = np.zeros((len(y_slices), len(all_iterations)))
        for index, ts in enumerate(all_iterations):
            np_data[:, index] = slice_emit[index][1:]
        ps = 1.e12  # for conversion from s to ps
        max_iter = max(all_iterations * dt * ps)
        # np_data.T * 1.e6 converts emittance to pi mm mrad,
        # y_slices * 1.e6 converts y slice position to micrometer
        self.plt_obj[idx] = self.ax.imshow(np_data.T * 1.e6, aspect="auto",
                                           norm=LogNorm(), origin="lower",
                                           vmin=1e-1, vmax=1e2,
                                           extent=(0, max(y_slices*1.e6),
                                                   0, max_iter),
                                           cmap=self.colors[idx])
        if self.cur_iteration:
            self.plt_lin = self.ax.axhline(self.cur_iteration * dt * ps,
                                           color='#FF6600')

        # create the colorbar and a separate ax for it
        self.colorbars[idx] = plt.colorbar(
            self.plt_obj[idx], cax=self.colorbar_axes[idx])
        self.colorbars[idx].solids.set_edgecolor("face")
        self.colorbars[idx].ax.text(
            .5, .5, self.sim_labels[idx], ha='center',
            va='center', rotation=270,
            transform=self.colorbar_axes[idx].transAxes)

    def _update_plt_obj(self, idx):
        """
        Implementation of base class function.
        """
        slice_emit, y_slices, all_iterations, dt = self.data[idx]
        np_data = np.zeros((len(y_slices), len(all_iterations)))
        for index, ts in enumerate(all_iterations):
            np_data[:, index] = slice_emit[index][1:]
        # np_data.T*1.e6 for conversion of emittance to pi mm mrad
        self.plt_obj[idx].set_data(np_data.T*1.e6)
        if self.plt_lin:
            self.plt_lin.remove()
        ps = 1.e12  # for conversion from s to ps
        if self.cur_iteration:
            self.plt_lin = self.ax.axhline(self.cur_iteration * dt * ps,
                                           color='#FF6600')
        self.plt_obj[idx].autoscale()
        self.colorbars[idx].update_normal(self.plt_obj[idx])

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
            species_filter: string
                name of the particle species filter, default is 'all'
                (defined in ``particleFilters.param``)

        """
        self.cur_iteration = kwargs.get('iteration')
        # passing iteration=None to your DataReader requests all iterations,
        # which is what we want here.
        kwargs['iteration'] = None
        super().visualize(**kwargs)

    def adjust_plot(self, **kwargs):
        species = kwargs['species']
        species_filter = kwargs.get('species_filter', 'all')
        idx = [
            i for i, cbar in enumerate(self.colorbars) if cbar is not None][0]
        self.colorbars[idx].ax.text(
            -1.2, 0.5, r'emittance [$\pi$ mm mrad]',
            ha='center', va='center',
            transform=self.colorbar_axes[idx].transAxes,
            rotation=270)
        self.ax.set_xlabel(r'y-slice [$\mathrm{\mu m}$]')
        self.ax.set_ylabel('time [ps]')
        self.ax.set_title('slice emittance for species ' +
                          species + ', filter = ' + species_filter)

    def clear_cbar(self):
        """Clear colorbars if present."""
        for idx in range(len(self.colorbars)):
            if self.colorbars[idx] is not None:
                # NOTE: maybe here get rid of the colorbars completely
                # by using self.colorbars[idx].remove() which removes
                # the cax (which equals self.colorbar_axes[idx]) from the
                # figure
                self._remove_colorbar(idx)


if __name__ == '__main__':

    def main():
        import sys
        import getopt

        def usage():
            print("usage:")
            print(
                "python", sys.argv[0],
                "-p <path to run_directory>"
                " -s <particle species> -f <species_filter> -i <iteration>")

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
        if path is None:
            print("Path to 'run' directory have to be provided!")
            usage()
            sys.exit(2)
        if species is None:
            species = 'e'
            print("Particle species was not given, will use", species)
        if filtr is None:
            filtr = 'all'
            print("Species filter was not given, will use", filtr)

        fig, ax = plt.subplots(1, 1)
        Visualizer(path, ax).visualize(iteration=iteration, species=species,
                                       species_filter=filtr)
        plt.show()

    main()
