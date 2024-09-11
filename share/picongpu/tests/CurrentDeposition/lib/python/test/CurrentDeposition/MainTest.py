"""
This file is part of PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Hannes Wolf
License: GPLv3+

This script reads the data provided by the PIConGPU simulation of the Current
Deposition test with the help of the openPMD-api.
The positions of the 0th and 1st iterations are the input for the reference
implementation grid_class.py.
The results of the current density field of both the reference and PIConGPU are
compared.
Returns whether they coincide or not.

"""


import openpmd_api as opmd
import numpy as np
from grid_class import grid
import sys


def get_params(path):
    """Creates a series of the data found at path.

    Parameters:
    path(string): path to the data to be compared against python reference.
    Path should have the form data_%T to provide access to all iterations.

    Returns:
    series: created series
    order(int): order of assignment function
    parameter list containing:
    charge(float): charge of the particle
    params(dict): simulation parameters(e.g. cell dimensions, unit_factors)

    """

    # load series
    series = opmd.Series(path, opmd.Access.read_only)
    indices = np.array(series.iterations)
    i = series.iterations[indices[0]]

    if len(series.iterations) == 1:
        raise ValueError("There is just 1 iteration in the series " "make sure, there are at least two")

    elif len(np.array(i.particles)) != 2:
        raise ValueError("There is more than one particle in the series " "make sure, there are exactly 2")

    else:
        # read the order of the assignment function
        i = series.iterations[0]
        electrons = i.particles["e"]
        order = int(electrons.get_attribute("particleShape"))

        params = {
            # read the parameters of the simulation
            "cell_depth": i.get_attribute("cell_depth"),
            "cell_height": i.get_attribute("cell_height"),
            "cell_width": i.get_attribute("cell_width"),
            "dt": i.get_attribute("dt"),
            "unit_time": i.get_attribute("unit_time"),
            "unit_length": i.get_attribute("unit_length"),
            "unit_charge": i.get_attribute("unit_charge"),
        }

        return series, order, params


def read_series(series):
    """Reads the position of the particle and the current density provided by
    the simulation as well as the charge of the particle.

    Parameters:
    series: simulation data

    Returns:
        Js(list): current density (contains an array of each component for each
                                   iteration)
        poss(list): relative in-cell positions of the particle in the series
        pos_offs(list): all node positions of the particle in the series
        charge(float): charge of particle

    """
    # create arrays for saving the data
    Js = []
    poss = []
    pos_offs = []

    for j in series.iterations:
        i = series.iterations[j]  # current iteration

        # current density field
        J_x = i.meshes["J"]["x"].load_chunk()
        J_y = i.meshes["J"]["y"].load_chunk()
        J_z = i.meshes["J"]["z"].load_chunk()

        # charge
        charge = i.particles["e"]["charge"][opmd.Mesh_Record_Component.SCALAR]

        # node coordinates
        pos2_x = i.particles["e"]["position"]["x"].load_chunk()
        pos2_y = i.particles["e"]["position"]["y"].load_chunk()
        pos2_z = i.particles["e"]["position"]["z"].load_chunk()

        # InCell coordinates
        pos_off2_x = i.particles["e"]["positionOffset"]["x"].load_chunk()
        pos_off2_y = i.particles["e"]["positionOffset"]["y"].load_chunk()
        pos_off2_z = i.particles["e"]["positionOffset"]["z"].load_chunk()

        series.flush()

        # write coordinate tuple
        pos2 = np.array([*pos2_x, *pos2_y, *pos2_z])
        pos_off = np.array([*pos_off2_x, *pos_off2_y, *pos_off2_z])

        # write current density tuple
        J = [J_x, J_y, J_z]
        Js.append(J)
        poss.append(pos2)
        pos_offs.append(pos_off)

    return Js, poss, pos_offs, charge


def compare(j_grid_x, j_grid_y, j_grid_z, J):
    """Compares the current density J provided by the simulation with the
    values j_grid_x/y/z from the python reference implementation.
    herefore abs(j(PIConGPU) - j(Python)) is calculated and compared against
    epsilon = 1e-5, which is slightly higher than PIConGPU numerical
    uncertainty.

    Parameters:
    j_grid_x/y/z(np.array): current density calculated by the reference
    J(np.array): current density calculated by PIConGPU

    Returns:
    0: if both current densities coincide
    42: if they don't coincide
    43: if the masked arrays _not_zero don't have the same shape

    """

    # compare the arrays values by subtracting
    x_compare = abs(J[0] - j_grid_x)
    y_compare = abs(J[1] - j_grid_y)
    z_compare = abs(J[2] - j_grid_z)

    # error boundary; slightly higher than numerical uncertainty
    epsilon = 1e-5

    # are the reference and simulated values the same? (within error margin)
    if np.all(x_compare < epsilon) and np.all(y_compare < epsilon) and np.all(z_compare < epsilon):
        print("simulation and reference coincide")
        return 0
    else:
        print("no consensus between simulation and reference")
        print("please check the simulation")
        return 42


def main(dataPath):
    """Main function.

    Parameters:
    dataPath(string): path to the simulation data

    Returns:
    compare_result(int): 0 if the simulated result can be regarded equal to the
                         reference
                         !0 if they differ.

    """
    # read parameters of series
    series, order, params = get_params(dataPath)

    # read current densities, charge, positions
    Js, poss, pos_offs, charge = read_series(series)
    # read the size of the current density array
    shape = np.shape(Js[1][0])

    # create grid_object for comparison
    compare_grid = grid(order)

    grid_x, grid_y, grid_z = compare_grid.create_grid()
    start_koord, end_koord = compare_grid.particle_step(poss[0], poss[1], pos_offs[0], pos_offs[1])
    # computation of current density by python reference implementation
    W_grid_x, W_grid_y, W_grid_z = compare_grid.current_deposition_field(start_koord, end_koord, grid_x, grid_y, grid_z)

    j_grid_x, j_grid_y, j_grid_z = compare_grid.current_density_field(
        W_grid_x, W_grid_y, W_grid_z, start_koord, end_koord, charge[0], params, shape
    )

    # comparison
    compare_result = compare(j_grid_x, j_grid_y, j_grid_z, Js[1])

    return compare_result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <path_to_simulation_data>")
    sys.exit(main(sys.argv[1]))
