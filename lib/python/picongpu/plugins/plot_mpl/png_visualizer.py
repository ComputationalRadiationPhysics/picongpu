"""
This file is part of the PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Visualizer(object):
    """
    Class for providing a plot of a PNG file using matplotlib.
    """

    def __init__(self, run_directory):
        """
        Parameters
        ----------
        run_directory: string
            path to the run directory of PIConGPU
            (the path before ``simOutput/``)
        """
        if run_directory is None:
            raise ValueError('The run_directory parameter can not be None!')

        self.run_directory = run_directory
        self.png_subdir = "simOutput/pngElectronsYX"
        self.plt_obj = None

    def get_data_path(self):
        return os.path.join(self.run_directory, self.png_subdir)

    def visualize(self, iteration, ax, **kwargs):
        """
        Creates a plot on the provided axes object for
        the PNG file of the given iteration using matpotlib.

        Parameters
        ----------
        iteration: int
            the iteration number for which data will be plotted.
        ax: matplotlib axes object
            the part of the figure where this plot will be shown.
        kwargs: dict
            possible additional keyword args (e.g. for styling).
            NOTE: no options from this parameter are considered yet!
        """
        # get the available png files
        png_path = self.get_data_path()
        # list with complete path for all png images
        png_files = [os.path.join(png_path, f)
                     for f in os.listdir(png_path) if f.endswith(".png")]

        # make sure they are in correct timely order
        png_files = sorted(png_files)

        # find the png file that matches the current iteration number
        img_file = ""
        for png in png_files:
            # take only filename without path (e.g. 'e_png_yx_0.5_001024.png')
            tmp = os.path.basename(png)
            # split the iteration number from the filename
            png_iter = int(tmp.split("_")[4].split(".")[0])
            if iteration == png_iter:
                img_file = png

        if img_file:
            img = mpimg.imread(img_file)

            # on first plotting command the object is created
            # afterwards, only data is updated
            if self.plt_obj is None:
                self.plt_obj = ax.imshow(img)
            else:
                self.plt_obj.set_data(img)

        else:
            print("Could not find png_file for iteration ", iteration)


if __name__ == '__main__':

    def main():

        import sys
        import getopt

        def usage():
            print("usage:")
            print(
                "python", sys.argv[0],
                "-p <path to run directory> -i <iteration>")

        path = None
        iteration = None

        try:
            opts, args = getopt.getopt(sys.argv[1:], "hp:i:", [
                "help", "path", "iteration"])
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

        # check that we got all args that we need
        if path is None or iteration is None:
            print("Path to 'run' directory and iteration have to be provided!")
            usage()
            sys.exit(2)

        fig, ax = plt.subplots(1, 1)
        Visualizer(path).visualize(iteration, ax)
        plt.show()

    main()
