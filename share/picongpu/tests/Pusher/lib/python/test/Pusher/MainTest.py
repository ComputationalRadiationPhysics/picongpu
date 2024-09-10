"""
This file is part of PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Hannes Wolf
License: GPLv3+

Test script to determine if a pusher works correctly. This is done by the
comparison of the radii of a particle at constant speed in a homogeneous
magnetic field. If the relative change between two periods is greater than
epsilon = 1e-5, the test fails. Also, the absolute phase change during one turn
is regarded. It should be smaller than 0.2 rad.
"""

import openpmd_api as opmd
import numpy as np
import sys


def get_params(path):
    """Creates a series of the data found at path.
    Reads the simulation parameters and saves them in the list params.

    Parameters:
    path (string): Path to the data to be compared against the Python
                   reference.
                   Path should have the form 'data_%T' to provide access to
                   all iterations.

    Returns:
    series: Created series
    order (int): Order of the assignment function
    parameter_list: List containing parameters:
        charge (float): Charge of the particle
        cell_depth/width/height (float): Cell dimensions (respectively)
        dt (float): Duration of a timestep
        unit_time (float): Conversion factor of arbitrary PIConGPU time units
                           to SI
        unit_charge (float): Conversion factor of arbitrary PIConGPU charge
                             units to SI
        unit_length (float): Conversion factor of arbitrary PIConGPU length
                             units to SI

    """
    # load series
    series = opmd.Series(path, opmd.Access.read_only)
    indices = np.array(series.iterations)
    i = series.iterations[indices[0]]

    if len(series.iterations) == 1:
        raise ValueError("There is just 1 iteration in the series" "make sure, there are at least two")

    elif len(np.array(i.particles)) != 1:
        raise ValueError("There is not only 1 particle in the series" "make sure, there is only one")

    else:
        # read parameters of the simulation
        params = {
            "cell_depth": i.get_attribute("cell_depth"),
            "cell_height": i.get_attribute("cell_height"),
            "cell_width": i.get_attribute("cell_width"),
            "dt": i.get_attribute("dt"),
            "unit_time": i.get_attribute("unit_time"),
            "unit_length": i.get_attribute("unit_length"),
            "unit_charge": i.get_attribute("unit_charge"),
            "unit_bfield": i.get_attribute("unit_bfield"),
            "unit_speed": i.get_attribute("unit_speed"),
            "unit_mass": i.get_attribute("unit_mass"),
        }

        return series, params


def read_series(series):
    """Reads the position and momentum of the particle in the x-y-plane
    provided by the simulation as well as the charge and mass of the particle.

    Parameters:
    series: Simulation data

    Returns:
    x/y_poss (np.array): Relative in-Cell positions of the particle in the
                         series
    x/y_offSet (np.array): All knot-positions of the particle in the series
    period (int): openPMD period of the simulation
    charge (float): Charge of particle
    mass (float): Mass of particle
    x/y_momentum (float): All momenta of the particle in the series
    R_c (float): Radius of gyration of the particle at timestep 0

    """
    num_iter = len(series.iterations)

    # create arrays for saving the data

    # in-cell position
    x_poss = np.zeros(num_iter)
    y_poss = np.zeros(num_iter)

    # cells
    x_offSet = np.zeros(num_iter)
    y_offSet = np.zeros(num_iter)

    # momentum
    x_momentum = np.zeros(num_iter)
    y_momentum = np.zeros(num_iter)

    i = series.iterations[0]
    # charge
    charge = i.particles["e"]["charge"][opmd.Mesh_Record_Component.SCALAR]
    # mass
    mass = i.particles["e"]["mass"][opmd.Mesh_Record_Component.SCALAR]

    # calculating of the openPMD-period for later use
    indices = np.array(series.iterations)
    period = indices[1] - indices[0]

    for iteration in indices:
        i = series.iterations[iteration]  # current position

        # node coordinates
        pos2_x = i.particles["e"]["position"]["x"].load_chunk()
        pos2_y = i.particles["e"]["position"]["y"].load_chunk()

        # InCell coordinates
        pos_off2_x = i.particles["e"]["positionOffset"]["x"].load_chunk()
        pos_off2_y = i.particles["e"]["positionOffset"]["y"].load_chunk()

        p_x = i.particles["e"]["momentum"]["x"].load_chunk()
        p_y = i.particles["e"]["momentum"]["y"].load_chunk()

        series.flush()

        array_index = int(iteration / period)
        # save the read data in arrays for later usage
        x_poss[array_index] = pos2_x
        y_poss[array_index] = pos2_y

        x_offSet[array_index] = pos_off2_x
        y_offSet[array_index] = pos_off2_y

        x_momentum[array_index] = p_x
        y_momentum[array_index] = p_y

    return (x_poss, y_poss, x_offSet, y_offSet, x_momentum, y_momentum, charge, mass, num_iter, period)


def compare_radius(
    x_poss, y_poss, x_offSet, y_offSet, x_momentum, y_momentum, charge, mass, B, params, num_iter, period
):
    """Tests if the change in radius from one revolution to the other is
    greater than epsilon = 1e-5.
    The radii are calculated by the positions of the particle in the x-y-plane
    in one test and by the momenta (and q*B) in the x-y-plane in the other one.

    Parameters:
    x/y_momentum(array): Series of the momenta of the particle in x/y-direction
    charge (float): Charge of the particle
    mass( float): Mass of the particle
    B (float): Magnitude of the magnetic field B in the test
    params (dict): Simulation parameters
    num_iter (int): Number of iterations in the series
    period (int): openPMD period of the simulation

    Returns:
    compare_result (int):
        == 0 if the radius of gyration does not vary much (<epsilon)
        != 0 if the change in radius of gyration large (>epsilon)

    """
    # the  change in of the radius, measured between 2 turns
    # is ca. 2e-5 (calculated with position)
    # Doubling this value yields epsilon as an approximation of the maximal
    # uncertanty PIConGPU should have when the test uses the position.
    # Therefore the test uses this value to test the radius change against.
    # PIConGPU run September 2023
    epsilon_position = 5e-5

    # calculating gamma for the calculation of the periodic time
    abs_momentum = np.sqrt(x_momentum[0] ** 2 + y_momentum[0] ** 2)
    gamma = 1 / np.sqrt(1 - abs_momentum**2 / (abs_momentum**2 + 1))
    # periodic time / timestep
    steps_per_revolution = (2 * np.pi * gamma * mass[0] / (abs(charge[0]) * B / params["unit_bfield"])) / params["dt"]

    revolutions_in_series = int(num_iter / steps_per_revolution)

    compare_result_momentum = 0
    compare_result_positions = 0

    # compare with position
    radius = np.sqrt((x_poss + x_offSet) ** 2 + (y_poss + y_offSet) ** 2)

    # relative comparison
    for k in range(revolutions_in_series):
        index_of_revolution = int(steps_per_revolution / period) * k
        compare_radii = abs(radius[index_of_revolution] - radius[index_of_revolution + 1]) / radius[index_of_revolution]

        if compare_radii > epsilon_position:
            print("pusher is not valid (position)")
            print("please check the simulation")
            compare_result_positions += 42

        else:
            print("pusher is valid (position)")

    # the greatest change in of the radius, measured between 2 turns
    # (2000 Turns simulated) is ca. 4e-6 (calculated with momentum)
    # Doubling this value yields epsilon as an approximation of the maximal
    # uncertanty PIConGPU should have when the test uses the momentum.
    # Therefore the test uses this value to test the radius change against.
    # PIConGPU run September 2023
    epsilon_momentum = 1e-5

    # comparision with momentum
    radius = (
        np.sqrt(x_momentum**2 + y_momentum**2)
        * params["unit_mass"]
        * params["unit_speed"]
        / (abs(charge[0]) * params["unit_charge"] * B)
        / params["cell_depth"]
        / params["unit_length"]
    )
    # relative comparison
    for k in range(revolutions_in_series):
        index_of_revolution = int(steps_per_revolution / period) * k
        compare_radii = abs(radius[index_of_revolution] - radius[index_of_revolution + 1]) / radius[index_of_revolution]

        if compare_radii > epsilon_momentum:
            print("pusher is not valid (position)")
            print("please check the simulation")
            compare_result_momentum += 46

        else:
            print("pusher is valid (momentum)")

    return compare_result_momentum + compare_result_positions


def compare_phases(
    x_poss, y_poss, x_offSet, y_offSet, x_momentum, y_momentum, charge, mass, B, params, num_iter, period
):
    """Tests if the phase difference from one revolution to the other is
    greater than delta = 0.25.
    The radii are calculated by the momenta (and q*B) in the x-y-plane.

    Parameters:
    x/y_momentum(array): Series of the momenta of the particle in x/y-direction
    charge (float): Charge of the particle
    mass (float): Mass of the particle
    B (float): Magnitude of the magnetic field B in the test
    params (dict): Simulation parameters
    num_iter (int): Number of iterations in the series
    period (int): openPMD period of the simulation

    Returns:
    0: if the phase difference is small (<delta)
    1: if the phase difference large (>delta)

    """
    # The measured phase difference between two revolutions is approximately
    # 0.08rad. Doubling this value represents the maximal error to be accepted
    # for the phase difference in the PIConGPU.
    # PIConGPU run of September 2023.
    delta = 0.16

    # calculating gamma for the calculation of the periodic time
    # (gamma factor for motion in the x-y plane)
    abs_momentum = np.sqrt(x_momentum[0] ** 2 + y_momentum[0] ** 2)
    gamma = 1 / np.sqrt(1 - abs_momentum**2 / (abs_momentum**2 + 1))
    # periodic time / timestep
    steps_per_revolution = (2 * np.pi * gamma * mass[0] / (abs(charge[0]) * B / params["unit_bfield"])) / params["dt"]

    revolutions_in_series = int(num_iter / steps_per_revolution)

    x = x_poss + x_offSet  # real x coordinates
    R_c = np.mean(
        (
            np.sqrt(x_momentum**2 + y_momentum**2)
            * params["unit_mass"]
            * params["unit_speed"]
            / (abs(charge[0]) * params["unit_charge"] * B)
            / params["cell_depth"]
            / params["unit_length"]
        )
    )

    theta = np.arccos(x / R_c)  # calculate the phase

    for k in range(revolutions_in_series):
        index_of_revolution = int(steps_per_revolution / period) * k
        # compare the phase after 1 periodic time
        compare_phases = abs(
            theta[index_of_revolution] - theta[index_of_revolution + int(steps_per_revolution / period)]
        )
        # compare with theoretical, i.e. stationary phase
        compare_ana_phases = abs(theta[index_of_revolution] - np.pi / 2 + 2 * np.pi / 100)

        compare_result_phases = 0

        if compare_phases > delta:
            print("pusher is not valid (phases)")
            print("please check the simulation")
            compare_result_phases += 1

        if compare_ana_phases > (k + 1) * delta:
            print("pusher is not valid (phases_analytical)")
            print("please check the simulation")
            compare_result_phases += 2

        else:
            print("pusher is valid (phases)")
    return compare_result_phases


def correct_starting_values_for_technical_details(x_poss, y_poss, x_offSet, y_offSet, R_c):
    """The particle is initialized at the arbitary position (5, 32) in the x-y-plane (some arbitrary z which is irrelevant for our computation...).
    So we have to shift the coordinates in a form, that the initialization point is exactly on the circle with the radius of the particle trajectory with its center at the origin.
    This shifting is done by the transformation (subtraction of the initialization coordinates and radius).
    But we also need to compensate for a half-step, which is automatically done by PIConGPU to improve the accuracy of the Boris Pusher
    (see: B. Ripperda et al 2018 ApJS 235 21 https://iopscience.iop.org/article/10.3847/1538-4365/aab114 ; 10.3847/1538-4365/aab114).
    This half-step is the first data written by PIConGPU and is corrected by the trigonometry operations in the transformation.

    """
    # the coordinates where the particle is initialized in the simulation
    initial_x_offSet = 32
    initial_y_offSet = 5

    x_poss = x_poss + (R_c * np.sin(2 / 100 * np.pi))
    y_poss = y_poss - R_c + int(R_c) + (R_c - R_c * np.cos(2 / 100 * np.pi))

    x_offSet = x_offSet - initial_x_offSet
    y_offSet = y_offSet - initial_y_offSet - int(R_c)

    return x_poss, y_poss, x_offSet, y_offSet


def main(dataPath):
    """Main function. Calls the functions to compare PIConGPU's simulation
    results with respect to the change of gyration radius and phase after
    one revolution.

    Returns:
    compare_result(int): 0 if the simulated result of phase and gyration radius
                         can be regarded as stationary
                         Non-zero if they diverge.

    """
    # read parameters of series
    series, params = get_params(dataPath)

    B = 50
    # extract all relevant data
    (x_poss, y_poss, x_offSet, y_offSet, x_momentum, y_momentum, charge, mass, num_iter, period) = read_series(series)

    # all radii in the series
    radius = (
        np.sqrt(x_momentum**2 + y_momentum**2)
        * params["unit_mass"]
        * params["unit_speed"]
        / (abs(charge[0]) * params["unit_charge"] * B)
        / params["cell_depth"]
        / params["unit_length"]
    )
    R_c = radius[0]  # original radius

    # transformation for the correct calculation of the radii by position
    x_poss, y_poss, x_offSet, y_offSet = correct_starting_values_for_technical_details(
        x_poss, y_poss, x_offSet, y_offSet, R_c
    )

    # tests
    compare_result_radius = compare_radius(
        x_poss, y_poss, x_offSet, y_offSet, x_momentum, y_momentum, charge, mass, B, params, num_iter, period
    )
    compare_result_phases = compare_phases(
        x_poss, y_poss, x_offSet, y_offSet, x_momentum, y_momentum, charge, mass, B, params, num_iter, period
    )

    # yield both tests/comparisions a positive result?
    compare_result = compare_result_radius + compare_result_phases

    return compare_result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"Usage: {sys.argv[0]} <path_to_simulation_data>")
    sys.exit(main(sys.argv[1]))
