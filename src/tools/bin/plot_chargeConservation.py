#!/usr/bin/env python
#
# Copyright 2015-2019 Richard Pausch
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

import argparse
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt

__doc__ = '''
This program reads electric field and charge density data
from hdf5 files created by PIConGPU and checks charge conservation
for the Yee scheme.

Three slice plots show the error in $div(E) - rho/epsilon_0$
normalized to the maximum [per-species] charge in the simulation.

Developer: Richard Pausch
'''


def set_colorbar(cb):
    """
    sets label and font size of color bar

    Parameters:
    cb: pyplot colorbar object
        colorbar to be adjusted
    """
    cb.set_label(r"$\vec \nabla \vec E - \rho / \varepsilon_0" +
                 r" \,[ \mathrm{max}(|\rho_k|) / \varepsilon_0 ]$",
                 fontsize=22)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(16)


def plotError(h5file, slice_pos=[0.5, 0.5, 0.5]):
    """
    read field data from hdf5 files
    compute div(E) - rho/epsilon_0
    plot slices through simulation volume

    Parameters:
    h5file: file name
        file name to hdf5 data set from PIConGPU

    slice_pos: list of floats
        list of 3 floats to define slice position [0, 1]
        Default=[0.5, 0.5, 0.5]
    """
    # load hdf5 file
    f = h5py.File(h5file, "r")

    # read time step (python 2 and 3 save)
    timestep = -1
    for i in f['/data'].keys():
        timestep = i

    # load physics constants and simulation parameters
    EPS0 = f["/data/{}".format(timestep)].attrs["eps0"]
    CELL_WIDTH = f["/data/{}".format(timestep)].attrs["cell_width"]
    CELL_HEIGHT = f["/data/{}".format(timestep)].attrs["cell_height"]
    CELL_DEPTH = f["/data/{}".format(timestep)].attrs["cell_depth"]

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
            species_Density = np.array(
                f["/data/{}/fields/".format(timestep) + field_name]
            )
            # choose norm to be the maximal charge density of all species
            norm = np.max([norm, np.amax(np.abs(species_Density))])
            # add charge density to total charge density
            charge += species_Density

    # close hdf5 file
    f.close()

    # compute divergence of electric field according to Yee scheme
    div = ((Ex[1:, 1:, 1:] - Ex[1:, 1:, :-1]) / CELL_WIDTH +
           (Ey[1:, 1:, 1:] - Ey[1:, :-1, 1:]) / CELL_HEIGHT +
           (Ez[1:, 1:, 1:] - Ez[:-1, 1:, 1:]) / CELL_DEPTH)

    # compute difference between electric field divergence and charge density
    diff = (div - charge[1:, 1:, 1:] / EPS0)

    limit = np.amax(np.abs(diff))

    # plot result
    plt.figure(figsize=(14, 5))

    plt.subplot(131)
    slice_cell_z = np.int(np.floor((diff.shape[0]-1) * slice_pos[0]))
    plt.title("slice in z at {}".format(slice_cell_z), fontsize=20)
    plt.imshow(diff[slice_cell_z, :, :],
               vmin=-limit, vmax=+limit,
               aspect='auto',
               cmap=plt.cm.bwr)
    plt.xlabel(r"$x\,[\Delta x]$", fontsize=20)
    plt.ylabel(r"$y\,[\Delta y]$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    set_colorbar(plt.colorbar(orientation='horizontal',
                              format="%2.2e", pad=0.18,
                              ticks=[-limit, 0, +limit])
                 )

    plt.subplot(132)
    slice_cell_y = np.int(np.floor((diff.shape[1]-1) * slice_pos[1]))
    plt.title("slice in y at {}".format(slice_cell_y), fontsize=20)
    plt.imshow(diff[:, slice_cell_y, :],
               vmin=-limit, vmax=+limit,
               aspect='auto',
               cmap=plt.cm.bwr)
    plt.xlabel(r"$x\,[\Delta x]$", fontsize=20)
    plt.ylabel(r"$z\,[\Delta z]$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    set_colorbar(plt.colorbar(orientation='horizontal',
                              format="%2.2e", pad=0.18,
                              ticks=[-limit, 0, +limit])
                 )

    plt.subplot(133)
    slice_cell_x = np.int(np.floor((diff.shape[2]-1) * slice_pos[2]))
    plt.title("slice in x at {}".format(slice_cell_x), fontsize=20)
    plt.imshow(diff[:, :, slice_cell_x],
               vmin=-limit, vmax=+limit,
               aspect='auto',
               cmap=plt.cm.bwr)
    plt.xlabel(r"$y\,[\Delta y]$", fontsize=20)
    plt.ylabel(r"$z\,[\Delta z]$", fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    set_colorbar(plt.colorbar(orientation='horizontal',
                              format="%2.2e", pad=0.18,
                              ticks=[-limit, 0, +limit])
                 )

    plt.tight_layout()

    if not args.output_file:
        plt.show()
    else:
        plt.savefig(args.output_file)


if __name__ == "__main__":
    # set up argument parser
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='For further questions please contact Richard Pausch.'
        )

    parser.add_argument(metavar="hdf5 file",
                        dest="h5file_name",
                        help='hdf5 file with PIConGPU data',
                        action='store',
                        type=str)

    parser.add_argument("--x",
                        dest="x_split",
                        action='store',
                        default=0.5,
                        type=np.float,
                        help='float value between [0,1] to set slice ' +
                             'position in x (default = 0.5)')

    parser.add_argument("--y",
                        dest="y_split",
                        action='store',
                        default=0.5,
                        type=np.float,
                        help='float value between [0,1] to set slice ' +
                             'position in y (default = 0.5)')

    parser.add_argument("--z",
                        dest="z_split",
                        action='store',
                        default=0.5,
                        type=np.float,
                        help='float value between [0,1] to set slice ' +
                             'position in z (default = 0.5)')

    parser.add_argument("--export",
                        metavar="file name",
                        dest="output_file",
                        default="",
                        help="export plot to file " +
                             "(disable interactive window)")

    args = parser.parse_args()

    slice_pos = np.clip([args.z_split,
                         args.y_split,
                         args.x_split],
                        0, 1)

    if os.path.isfile(args.h5file_name):
        plotError(args.h5file_name, slice_pos=slice_pos)
    else:
        print("ERROR: {} is not a file".format(args.h5file_name))
