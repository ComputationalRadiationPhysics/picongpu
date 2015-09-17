#!/usr/bin/env python
#
# Copyright 2015 Richard Pausch
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

__doc__ = """
This program reads electric field and charge density data
from all hdf5 files created by a PIConGPU simulation and
plots a variety of values to check charge conservation
over time.

All plotted values show the difference $d = div(E) - rho/epsilon_0$
normalized to the maximum [per-species] charge in the first 
simulation time step.

Developer: Richard Pausch
"""

# ---- function definitions ----------

# get all hdf5 output files
def get_list_of_hdf5_files(base_directory):
    """
    Returns a list of hdf5 files (`h5_*.h5`)
    listed in sub-directory `simOutput`

    Parameters:
    base_directory: string
        directory path where to find simOutput

    Return:
    list of strings with hdf5 file names found
    """
    h5_list = [] # empty list for hdf5 files
    h5_dir = base_directory + "/simOutput/"
    if not os.path.isdir(h5_dir):
        raise Exception(("Error: {} does not contain"+
                         " a simOutput directory").format(directory))

    for filename in os.listdir(h5_dir):
        if os.path.isfile(h5_dir+filename):
            if re.search("h5_.*.h5", filename):
                h5_list.append(h5_dir + filename)
    return h5_list


def deviation_charge_conservation(h5file):
    """
    read field data from hdf5 files
    compute d = div(E) - rho/epsilon_o0

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
    CELL_WIDTH = f["/data/{}".format(timestep)].attrs["cell_width"]
    CELL_HEIGHT = f["/data/{}".format(timestep)].attrs["cell_height"]
    CELL_DEPTH = f["/data/{}".format(timestep)].attrs["cell_depth"]
    CELL_VOLUME = CELL_WIDTH * CELL_HEIGHT * CELL_DEPTH

    # load electric field
    Ex = np.array(f["/data/{}/fields/FieldE/x".format(timestep)])
    Ey = np.array(f["/data/{}/fields/FieldE/y".format(timestep)])
    Ez = np.array(f["/data/{}/fields/FieldE/z".format(timestep)])

    # load and add charge density
    charge = np.zeros_like(Ex)
    norm = 0.0
    for field_name in f["/data/{}/fields/".format(timestep)].keys():
        if field_name[0:8] == "Density_":
            # load species density
            species_Density = np.array(f["/data/{}/fields/".format(timestep) + field_name])
            # choose norm to be the maximal charge density of all species
            norm = np.max([norm, np.amax(np.abs(species_Density))])
            # add charge density to total charge density
            charge += species_Density

    # close hdf5 file
    f.close()

    # compute divergence of electric field according to Yee scheme
    div = ((Ex[1:, 1:, 1:] - Ex[1:, 1:, :-1])/CELL_WIDTH +
           (Ey[1:, 1:, 1:] - Ey[1:, :-1, 1:])/CELL_HEIGHT +
           (Ez[1:, 1:, 1:] - Ez[:-1, 1:, 1:])/CELL_DEPTH)

    # compute difference between electric field divergence and charge density
    diff = (div  - charge[1:, 1:, 1:]/EPS0)

    return float(timestep), np.amax(np.abs(diff)), np.mean(np.abs(diff)), np.std(diff), norm


# ---- main program ----------

if __name__ == "__main__":
    # set up argument parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="For further questions please contact Richard Pausch."
        )

    parser.add_argument(metavar="simulation directory",
                        dest="directory",
                        help="simulation base directory",
                        action="store")

    args = parser.parse_args()
    directory = args.directory

    # do the data reading and catch errors
    try:
        # test if directory is a directory
        if not os.path.isdir(directory):
            raise Exception("Error: {} is not a directory".format(directory))

        # check if any hdf5 files were found
        h5_file_list = get_list_of_hdf5_files(directory)
        if len(h5_file_list) == 0:
            raise Exception("No hdf5 files found in {}".format(directory+"simOutput/"))

    except Exception as error_msg:
        print("{}".format(error_msg))
        sys.exit(1)

    # collect data from all found hdf5 files
    collect_results = None
    print("Read files:")
    for f in h5_file_list:
        print(f)
        t, cc_max, mean_abs, std, norm =  deviation_charge_conservation(f)
        data_tmp = np.array([[t, cc_max, mean_abs, std, norm]])
        if type(collect_results) == type(None):
            collect_results = data_tmp
        else:
            collect_results = np.append(collect_results, data_tmp, axis=0)

    # sort data temporally
    collect_results = np.sort(collect_results, axis=0)

    # alias to data
    t = collect_results[:, 0] # all timesteps
    max_diff = collect_results[:, 1] # all max abs diff
    mean_abs = collect_results[:, 2] # all mean abs
    std = collect_results[:, 3] # all std
    norm = collect_results[0, 4] # first (t=0) norm

    # plot data
    plt.figure(figsize=(10,5))
    plt.title(directory, fontsize=22)

    plt.subplot(121)
    plt.plot(t, max_diff/norm,
             linestyle="-",lw=3,
             marker="+", ms=15, markeredgewidth=3, color="blue")

    plt.xlabel("time step", fontsize=20)
    plt.ylabel("max(abs(d))/norm", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.subplot(122)
    plt.errorbar(t, mean_abs/norm, yerr=std/norm, lw=3, markeredgewidth=3, color="blue")
    plt.xlabel("time step", fontsize=20)
    plt.ylabel("mean(abs(d))/norm +/- std(d)/norm", fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()
