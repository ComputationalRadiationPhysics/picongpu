"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

from picongpu.plugins.data import PNGData
from picongpu.plugins.plot_mpl.base_visualizer import Visualizer as\
    BaseVisualizer


class Visualizer(BaseVisualizer):
    """
    Class for providing a plot of a PNG file using matplotlib.
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
        super().__init__(PNGData, run_directories, ax)

    def _check_and_fix_run_dirs(self, run_directories):
        """
        Overridden from base class. Makes sure to only accept
        a single simulation's run_directory.
        """
        base_checked = super()._check_and_fix_run_dirs(run_directories)

        # Fail if more than one run_directory since plotting
        # several PNGs at the same time does not make sense!
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
        self.plt_obj[idx] = self.ax.imshow(self.data[idx])

    def _update_plt_obj(self, idx):
        """
        Implementation of base class function.
        """
        self.plt_obj[idx].set_data(self.data[idx])

    def visualize(self, **kwargs):
        """
        Creates a plot on the provided axes object for
        the PNG file of the given iteration using matpotlib.

        Parameters
        ----------
        kwargs: dict
            additional keyword args. Necessary are the following:
            species : string
                short name of the particle species, e.g. 'e' for electrons
                (defined in ``speciesDefinition.param``)
            species_filter: string
                name of the particle species filter, default is 'all'
                (defined in ``particleFilters.param``)
            axis: string
                the coordinate system axis labels (e.g. 'yx' or 'yz')
            slice_point: float
                relative offset in the third axis not given in the axis\
                argument.\
                Should be between 0 and 1
            iteration: int or list of ints
                The iteration at which to read the data.
                if set to 'None', then return images for all available\
                    iterations
            time: float
                simulation time.
                Only one of 'iteration' or 'time' should be passed!
        """
        super().visualize(**kwargs)


if __name__ == '__main__':

    def main():

        import sys
        import getopt
        import matplotlib.pyplot as plt

        def usage():
            print("usage:")
            print(
                "python", sys.argv[0],
                "-p <path to run directory> -i <iteration>"
                " -s <particle species> -f <species filter>"
                " -a <axis> -o <slice point offset>")

        path = None
        iteration = None
        species = None
        filtr = None
        axis = None
        slice_point = None

        try:
            opts, args = getopt.getopt(sys.argv[1:], "hp:i:s:f:a:o:", [
                "help", "path", "iteration", "species", "filter", "axis",
                "offset"])
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
            elif opt in ["-a", "--axis"]:
                axis = arg
            elif opt in ["-o", "--offset"]:
                slice_point = float(arg)

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
        if axis is None:
            axis = "yx"
            print("Axis was not given, will use", axis)
        if slice_point is None:
            print("Offset was not given, will determine from file")

        _, ax = plt.subplots(1, 1)
        Visualizer(path, ax).visualize(iteration=iteration, species=species,
                                       species_filter=filtr, axis=axis,
                                       slice_point=slice_point)
        plt.show()

    main()
