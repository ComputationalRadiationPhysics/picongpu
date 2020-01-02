"""
This file is part of the PIConGPU.

Copyright 2017-2020 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from picongpu.plugins.data import PhaseSpaceData
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as\
    BaseVisualizer
from picongpu.plugins.plot_mpl.utils import get_different_colormaps


class Visualizer(BaseVisualizer):
    """
    Class for creating a matplotlib plot of phase space diagrams.
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
        # TODO: remove this and get it from the readers metadata
        # since it seems only correct for species='e' anyway.
        self.mu = 1.e6
        self.e_mc_r = 1. / (9.1e-31 * 2.9979e8)

        self.colorbars = None
        # the separate colorbar axes
        self.colorbar_axes = None

        super().__init__(PhaseSpaceData, run_directories, ax)

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
        # self.colorbars[idx].remove()
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
        Turns 'self.plt_obj' into a matplotlib.image.AxesImage object.
        """
        dat, meta = self.data[idx]
        self.plt_obj[idx] = self.ax.imshow(
            np.abs(dat).T * meta.dV,
            extent=meta.extent * [self.mu, self.mu, self.e_mc_r, self.e_mc_r],
            interpolation='nearest',
            aspect='auto',
            origin='lower',
            norm=LogNorm(),
            cmap=self.colors[idx])
        # create the colorbar and a separate ax for it
        self.colorbars[idx] = plt.colorbar(
            self.plt_obj[idx], cax=self.colorbar_axes[idx])
        self.colorbars[idx].solids.set_edgecolor("face")
        self.colorbars[idx].ax.text(
            .5, .5, self.sim_labels[idx], ha='center',
            va='center', rotation=270,
            transform=self.colorbar_axes[idx].transAxes)

        self.ax.set_xlabel(r'${0}$ [${1}$]'.format(meta.r, r'\mathrm{\mu m}'))
        self.ax.set_ylabel(r'$p_{0}$ [$\beta\gamma$]'.format(meta.p))

    def _update_plt_obj(self, idx):
        """
        Implementation of base class function.
        """
        dat, meta = self.data[idx]
        self.plt_obj[idx].set_data(np.abs(dat).T * meta.dV)
        self.plt_obj[idx].autoscale()
        # update the colorbar
        self.colorbars[idx].update_normal(self.plt_obj[idx])

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

    def adjust_plot(self, **kwargs):
        """Overridden from base."""
        # only for the first index that is not None we set the description
        # (which is the innermost colorbar)
        idx = [
            i for i, cbar in enumerate(self.colorbars) if cbar is not None][0]

        self.colorbars[idx].ax.text(
            -1.2, 0.5,
            r'$Q / \mathrm{d}r \mathrm{d}p$ [$\mathrm{C s kg^{-1} m^{-2}}$]',
            ha='center', va='center',
            transform=self.colorbar_axes[idx].transAxes,
            rotation=270)

        # prevent squeezing of colorbars and labels
        self.ax.figure.tight_layout()

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
