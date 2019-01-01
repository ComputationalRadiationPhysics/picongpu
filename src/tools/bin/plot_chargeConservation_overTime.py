#!/usr/bin/env python
#
# Copyright 2015-2019 Richard Pausch, Axel Huebl
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

# import system interface modules
import os
import re
import sys
import argparse

# import data analysis and plotting modules
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter

__doc__ = """
This program reads electric field and charge density data
from all hdf5 files created by a PIConGPU simulation and
plots a variety of values to check charge conservation
over time.

All plotted values show the difference $d = div(E)*epsilon_0 - rho$
normalized to the maximum [per-species] charge in the first
simulation time step.

Developer: Richard Pausch
"""


def get_list_of_hdf5_files(base_directory):
    """
    Returns a list of hdf5 files (`*_<step>.h5`)
    listed in sub-directory `simOutput/h5/`

    Parameters:
    base_directory: string
        directory path where to find simOutput/h5/

    Return:
    list of strings with hdf5 file names found
    """
    h5_list = []  # empty list for hdf5 files
    h5_dir = base_directory + "/simOutput/h5/"
    if not os.path.isdir(h5_dir):
        raise Exception(("Error: {} does not contain" +
                         " a simOutput/h5/ directory").format(directory))

    for filename in os.listdir(h5_dir):
        if os.path.isfile(h5_dir+filename):
            if re.search(r".+_[0-9]+\.h5", filename):
                h5_list.append(h5_dir + filename)
    return h5_list


def deviation_charge_conservation(h5file):
    """
    read field data from hdf5 files
    compute d = div(E)*epsilon_0 - rho

    Parameters:
    h5file: file name
        file name and path to hdf5 data from PIConGPU

    Return:
    list of floats: [timestep, max(abs(d)),
        mean(abs(d)), std(d), norm]
    """
    # load hdf5 file
    f = h5py.File(h5file, "r")

    # read time step (python 2 and 3 save)
    timestep = -1
    for i in f["/data"].keys():
        timestep = i

    # load physics constants and simulation parameters
    EPS0 = f["/data/{}".format(timestep)].attrs["eps0"]
    is2D = False

    # load electric field
    Ex = np.array(f["/data/{}/fields/E/x".format(timestep)])
    Ey = np.array(f["/data/{}/fields/E/y".format(timestep)])
    Ez = np.array(f["/data/{}/fields/E/z".format(timestep)])

    # load and add charge density
    charge = np.zeros_like(Ex)
    norm = 0.0
    for field_name in f["/data/{}/fields/".format(timestep)].keys():
        if field_name[-14:] == "_chargeDensity":
            # load species density
            species_Density_pointer = f["/data/{}/fields/".format(timestep) +
                                        field_name]
            species_Density = np.array(species_Density_pointer)
            # choose norm to be the maximal charge density of all species
            norm = np.max([norm, np.amax(np.abs(species_Density))])
            # add charge density to total charge density
            charge += species_Density
            # We check the attribute _size of any/all density_[species].
            # libSplash always keeps this as an array of length 3. It
            # describes the size of the data in each dimension. If we are in
            # a 2D simulation, the size of the z or [2]-component is 1, which
            # is <2. The code changes the 2D3D flag if one Density data set is
            # 2D.
            if species_Density_pointer.attrs['_size'][2] < 2:
                is2D = True

    # load cell size and compute cell volume
    CELL_WIDTH = f["/data/{}".format(timestep)].attrs["cell_width"]
    CELL_HEIGHT = f["/data/{}".format(timestep)].attrs["cell_height"]
    CELL_DEPTH = f["/data/{}".format(timestep)].attrs["cell_depth"]

    # close hdf5 file
    f.close()

    if is2D:
        # compute divergence of electric field according to Yee scheme
        div = ((Ex[1:, 1:] - Ex[1:, :-1]) / CELL_WIDTH +
               (Ey[1:, 1:] - Ey[:-1, 1:]) / CELL_HEIGHT)

        # compute difference between electric field divergence and charge
        # density
        diff = (div * EPS0 - charge[1:, 1:])

    else:
        # compute divergence of electric field according to Yee scheme
        div = ((Ex[1:, 1:, 1:] - Ex[1:, 1:, :-1]) / CELL_WIDTH +
               (Ey[1:, 1:, 1:] - Ey[1:, :-1, 1:]) / CELL_HEIGHT +
               (Ez[1:, 1:, 1:] - Ez[:-1, 1:, 1:]) / CELL_DEPTH)

        # compute difference between electric field divergence and charge
        # density
        diff = (div * EPS0 - charge[1:, 1:, 1:])

    return float(timestep), np.amax(np.abs(diff)), np.mean(np.abs(diff)), \
        np.std(diff), norm


# ---- main program ----------

if __name__ == "__main__":
    # set up argument parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="For further questions please contact Richard Pausch."
        )

    parser.add_argument(metavar="simulation directories",
                        dest="directories",
                        help="simulation base directories",
                        action="store",
                        nargs="+")

    parser.add_argument("--export",
                        metavar="file name",
                        dest="output_file",
                        default="",
                        help="export plot to file " +
                             "(disable interactive window)")

    args = parser.parse_args()
    directories = args.directories

    # prepare plot of data
    plt.figure(figsize=(10, 5))
    plt.title("charge conservation over time", fontsize=22)

    major_locator1 = LinearLocator()
    major_locator2 = LinearLocator()
    major_formatter = FormatStrFormatter('%1.1e')

    ax1 = plt.subplot(121)
    ax1.set_xlabel(r"$t\,[\Delta t]$", fontsize=20)
    ax1.set_ylabel(
        r"$\mathrm{max}|d|\,[\rho_\mathrm{max}(0)]$",
        fontsize=20
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # always use scientific notation
    ax1.yaxis.set_major_locator(major_locator1)
    ax1.yaxis.set_major_formatter(major_formatter)

    ax2 = plt.subplot(122)
    ax2.set_xlabel(r"$t\,[\Delta t]$", fontsize=20)
    ax2.set_ylabel(
        r"$\left<|d|\right> \pm \sigma_d\,[\rho_\mathrm{max}(0)]$",
        fontsize=20
    )
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # always use scientific notation
    ax2.yaxis.set_major_locator(major_locator2)
    ax2.yaxis.set_major_formatter(major_formatter)

    # counter for simulation directories (avoids pyplot bug with
    # underscore labels)
    sim_dir_counter = 1

    for directory in directories:
        # do the data reading and catch errors
        try:
            # test if directory is a directory
            if not os.path.isdir(directory):
                raise Exception("Error: {} is not a directory".format(
                                directory))

            # check if any hdf5 files were found
            h5_file_list = get_list_of_hdf5_files(directory)
            if len(h5_file_list) == 0:
                raise Exception("No hdf5 files found in {}".format(
                                directory + "simOutput/h5/"))

        except Exception as error_msg:
            print("{}".format(error_msg))
            sys.exit(1)

        # collect data from all found hdf5 files
        collect_results = None
        print("Read files:")
        for f in h5_file_list:
            print(f)
            t, cc_max, mean_abs, std, norm = deviation_charge_conservation(f)
            data_tmp = np.array([[t, cc_max, mean_abs, std, norm]])
            if collect_results is None:
                collect_results = data_tmp
            else:
                collect_results = np.append(collect_results, data_tmp, axis=0)

        # sort data temporally
        collect_results = np.sort(collect_results, axis=0)

        # alias to data
        t = collect_results[:, 0]  # all timesteps
        max_diff = collect_results[:, 1]  # all max abs diff
        mean_abs = collect_results[:, 2]  # all mean abs
        std = collect_results[:, 3]  # all std
        norm = collect_results[0, 4]  # first (t=0) norm

        # generate plot label based on directory and avoid underscore bug
        plot_label = ("{:d}. ".format(sim_dir_counter) +
                      os.path.normpath(directory).split("/")[-1])
        sim_dir_counter += 1

        # add plot for maximum difference
        ax1.plot(t, max_diff/norm,
                 linestyle="-", lw=3,
                 marker="+", ms=15, markeredgewidth=3,
                 label=plot_label)

        # add plot for mean difference and std
        ax2.errorbar(t, mean_abs/norm, yerr=std/norm, lw=3, markeredgewidth=3,
                     label=plot_label)

    # finish plots
    ax1.legend(loc=0)
    ax2.legend(loc=0)
    plt.tight_layout()

    if not args.output_file:
        plt.show()
    else:
        plt.savefig(args.output_file)
