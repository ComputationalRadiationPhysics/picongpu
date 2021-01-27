#!/usr/bin/env python3
#
# Copyright 2015-2021 Richard Pausch
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
import numpy as np
import matplotlib.pyplot as plt
import openpmd_api as io

__doc__ = '''
This program reads electric field and charge density data
from openPMD files created by PIConGPU and checks charge conservation
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


def plotError(file_pattern, slice_pos=[0.5, 0.5, 0.5], timestep=-1):
    """
    read field data from an openPMD file
    compute div(E) - rho/epsilon_0
    plot slices through simulation volume

    Parameters:
    file_pattern: file name
         openPMD file series pattern e.g. simData_%%T.bp

    slice_pos: list of floats
        list of 3 floats to define slice position [0, 1]
        Default=[0.5, 0.5, 0.5]

    timestep: selected timestep
        simulation step used if file is an
        openPMD file series pattern e.g. simData_%%T.bp
    """
    # load file
    series = io.Series(file_pattern, io.Access.read_only)

    # read time step
    if timestep == -1:
        *_, timestep = series.iterations

    f = series.iterations[timestep]

    # load physics constants and simulation parameters
    EPS0 = f.get_attribute("eps0")
    CELL_WIDTH = f.get_attribute("cell_width")
    CELL_HEIGHT = f.get_attribute("cell_height")
    CELL_DEPTH = f.get_attribute("cell_depth")

    # load electric field
    Ex = f.meshes["E"]["x"][:]
    Ey = f.meshes["E"]["y"][:]
    Ez = f.meshes["E"]["z"][:]

    series.flush()

    # load and add charge density
    charge = np.zeros_like(Ex)
    norm = 0.0

    for fieldName in f.meshes:
        search_pattern = "_chargeDensity"
        if fieldName[-len(search_pattern):] == search_pattern:
            # load species density
            species_Density = \
                f.meshes[fieldName][io.Mesh_Record_Component.SCALAR][:]
            series.flush()
            # choose norm to be the maximal charge density of all species
            norm = np.max([norm, np.amax(np.abs(species_Density))])
            # add charge density to total charge density
            charge += species_Density

    # close file
    del series

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
    slice_cell_z = np.int(np.floor((diff.shape[0] - 1) * slice_pos[0]))
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
    slice_cell_y = np.int(np.floor((diff.shape[1] - 1) * slice_pos[1]))
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
    slice_cell_x = np.int(np.floor((diff.shape[2] - 1) * slice_pos[2]))
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

    parser.add_argument(metavar="openPMD file name",
                        dest="filename",
                        help='openPMD file or series pattern '
                             'with PIConGPU data',
                        action='store',
                        type=str)

    parser.add_argument("-t",
                        dest="selected_timestep",
                        help='simulation step used if file is an '
                             'openPMD file series pattern e.g. simData_%%T.bp',
                        action='store',
                        default=-1,
                        type=int)

    parser.add_argument("-x",
                        dest="x_split",
                        action='store',
                        default=0.5,
                        type=np.float,
                        help='float value between [0,1] to set slice ' +
                             'position in x (default = 0.5)')

    parser.add_argument("-y",
                        dest="y_split",
                        action='store',
                        default=0.5,
                        type=np.float,
                        help='float value between [0,1] to set slice ' +
                             'position in y (default = 0.5)')

    parser.add_argument("-z",
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

    plotError(args.filename, slice_pos=slice_pos,
              timestep=args.selected_timestep)
