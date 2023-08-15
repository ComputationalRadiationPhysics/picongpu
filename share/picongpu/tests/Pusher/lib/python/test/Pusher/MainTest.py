"""
This file is part of the PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Hannes Wolf
License: GPLv3+

Test script to determine if a pusher works correctly. This is done by the
comparison of the radii of a particle at constant speed in a homogeneous
magnetic field. If the relative change between two periods is greater than
epsilon = 1e-5, the test fails. Also, the absolute phase change during one turn
is regarded. It should be smaller than 0.2 rad.
"""

import openpmd_api as io
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
    series = io.Series(path, io.Access.read_only)
    indices = np.array(series.iterations)
    i = series.iterations[indices[0]]

    if len(series.iterations) == 1:
        raise ValueError("There is just 1 iteration in the series"
                         "make sure, there are at least two")

    elif len(np.array(i.particles)) != 1:
        raise ValueError("There is just 1 iteration in the series"
                         "make sure, there are at least two")

    else:
        # read parameters of the simulation
        cell_depth = i.get_attribute("cell_depth")
        cell_height = i.get_attribute("cell_height")
        cell_width = i.get_attribute("cell_width")
        dt = i.get_attribute("dt")
        unit_time = i.get_attribute("unit_time")
        unit_length = i.get_attribute("unit_length")
        unit_charge = i.get_attribute("unit_charge")
        unit_bfield = i.get_attribute("unit_bfield")
        unit_speed = i.get_attribute("unit_speed")
        unit_mass = i.get_attribute("unit_mass")

        return series, [
            cell_depth, cell_height, cell_width, dt, unit_time, unit_charge,
            unit_length, unit_mass, unit_speed, unit_bfield
            ]


def read_series(series):
    """Reads the position and momentum of the particle in the x-y-plane
    provided by the simulation as well as the charge and mass of the particle.

    Parameters:
    series: Simulation data

    Returns:
    x/y_poss (np.array): Relative in-Cell positions of the particle in the
                         series
    x/y_offSet (np.array): All knot-positions of the particle in the series
    periode (int): openPMD periode of the simulation
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
    charge = i.particles["e"]["charge"][io.Mesh_Record_Component.SCALAR]
    # mass
    mass = i.particles["e"]["mass"][io.Mesh_Record_Component.SCALAR]

    # calculating of the openPMD-periode for later use
    indices = np.array(series.iterations)
    periode = indices[1] - indices[0]

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

        array_index = int(iteration/periode)
        # save the read data in arrays for later usage
        x_poss[array_index] = pos2_x
        y_poss[array_index] = pos2_y

        x_offSet[array_index] = pos_off2_x
        y_offSet[array_index] = pos_off2_y

        x_momentum[array_index] = p_x
        y_momentum[array_index] = p_y

    return (x_poss, y_poss, x_offSet, y_offSet, x_momentum, y_momentum,
            charge, mass, num_iter, periode)


def compare_momentum(x_momentum, y_momentum,
                     charge, mass, B, params, num_iter, periode):
    """Tests if the change in radius from one revolution to the other is
    greater than epsilon = 1e-5.
    The radii are calculated by the momenta (and q*B) in the x-y-plane.

    Parameters:
    x/y_momentum(array): Series of the momenta of the particle in x/y-direction
    charge (float): Charge of the particle
    mass (float): Mass of the particle
    B (float): Magnitude of the magnetic field B in the test
    params (list): Simulation parameters
    num_iter (int): Number of iterations in the series
    period (int): openPMD period of the simulation

    Returns:
    0: if the radius of gyration does not vary much (<epsilon)
    42: if the change in radius of gyration large (>epsilon)

    """
    # the greatest change in of the radius, measured between 2 turns
    # (2000 Turns simulated) is ca. 4e-6. Doubling this value yields epsilon
    # as an approximation of the maximal uncertanty PIConGPU should have.
    # Therefore the test uses this value to test the radius change against
    # PIConGPU run September 2023
    epsilon = 1e-5

    # calculating the value of gamma for the determination of the periodic time
    abs_momentum = np.sqrt(x_momentum[0]**2 + y_momentum[0]**2)
    gamma = 1/np.sqrt(1 - abs_momentum**2/(abs_momentum**2 + 1))
    # periodic time / timestep
    steps_per_revolution = ((2*np.pi * gamma * mass[0]/(abs(charge[0])
                            * B/params[9])) / params[3])

    revolutions_in_series = int(num_iter / steps_per_revolution)

    radius = (np.sqrt(x_momentum**2 + y_momentum**2) * params[7] * params[8] /
              (abs(charge[0]) * params[5] * B) / params[0] / params[6])
    # relative comparison
    for k in range(revolutions_in_series):
        index_of_revolution = int(steps_per_revolution/periode) * k
        compare_radii = (
            abs(radius[index_of_revolution] - radius[index_of_revolution + 1])
            / radius[index_of_revolution])

        if compare_radii > epsilon:
            print("pusher is not valid (position)")
            print("please check the simulation")
            return 42

    print("pusher is valid (momentum)")
    return 0


def compare_positions(x_poss, y_poss, x_offSet, y_offSet,
                      x_momentum, y_momentum,
                      charge, mass, B, params, num_iter, periode):
    """Tests if the change in radius from one revolution to the other is
    greater than epsilon = 1e-5.
    The radii are calculated by the positions of the particle in the x-y-plane.

    Parameters:
    x/y_momentum(array): Series of the momenta of the particle in x/y-direction
    charge (float): Charge of the particle
    mass( float): Mass of the particle
    B (float): Magnitude of the magnetic field B in the test
    params (list): Simulation parameters
    num_iter (int): Number of iterations in the series
    period (int): openPMD period of the simulation

    Returns:
    0: if the radius of gyration does not vary much (<epsilon)
    43: if the change in radius of gyration large (>epsilon)

    """
    # the greatest change in of the radius, measured between 2 turns
    # (2000 Turns simulated) is ca. 4e-6. Doubling this value yields epsilon
    # as an approximation of the maximal uncertanty PIConGPU should have.
    # Therefore the test uses this value to test the radius change against
    # PIConGPU run September 2023
    epsilon = 1e-5

    # calculating gamma for the calculation of the periodic time
    abs_momentum = np.sqrt(x_momentum[0]**2 + y_momentum[0]**2)
    gamma = 1/np.sqrt(1 - abs_momentum**2/(abs_momentum**2 + 1))
    # periodic time / timestep
    steps_per_revolution = ((2*np.pi * gamma * mass[0]/(abs(charge[0])
                            * B/params[9])) / params[3])

    revolutions_in_series = int(num_iter / steps_per_revolution)

    radius = np.sqrt((x_poss + x_offSet)**2 + (y_poss + y_offSet)**2)

    # relative comparison
    for k in range(revolutions_in_series):
        index_of_revolution = int(steps_per_revolution/periode) * k
        compare_radii = (
            abs(radius[index_of_revolution] - radius[index_of_revolution + 1])
            / radius[index_of_revolution])

        if compare_radii > epsilon:
            print("pusher is not valid (position)")
            print("please check the simulation")
            return 43

    print("pusher is valid (position)")
    return 0


def compare_phases(x_poss, y_poss, x_offSet, y_offSet,
                   x_momentum, y_momentum,
                   charge, mass, B, params, num_iter, periode):
    """Tests if the phase difference from one revolution to the other is
    greater than delta = 0.25.
    The radii are calculated by the momenta (and q*B) in the x-y-plane.

    Parameters:
    x/y_momentum(array): Series of the momenta of the particle in x/y-direction
    charge (float): Charge of the particle
    mass (float): Mass of the particle
    B (float): Magnitude of the magnetic field B in the test
    params (list): Simulation parameters
    num_iter (int): Number of iterations in the series
    period (int): openPMD period of the simulation

    Returns:
    0: if the phase difference is small (<delta)
    1: if the phase difference large (>delta)

    """
    # The measured phase difference between two revolutions is approximately
    # 0.125. Doubling this value represents the maximal error to be accepted
    # for the phase difference in the PIConGPU.
    # PIConGPU run of September 2023.
    delta = 0.25

    # calculating gamma for the calculation of the periodic time
    # (gamma factor for motion in the x-y plane)
    abs_momentum = np.sqrt(x_momentum[0]**2 + y_momentum[0]**2)
    gamma = 1/np.sqrt(1 - abs_momentum**2/(abs_momentum**2 + 1))
    # periodic time / timestep
    steps_per_revolution = ((2*np.pi * gamma * mass[0]/(abs(charge[0])
                            * B/params[9])) / params[3])

    revolutions_in_series = int(num_iter / steps_per_revolution)

    x = x_poss + x_offSet     # real x coordinates
    R_c = np.mean(np.sqrt(x_momentum**2 + y_momentum**2) * params[7]
                  * params[8] / (abs(charge[0])*params[5] * B) / params[0] /
                  params[6])
    theta = np.arccos(x/R_c)  # calculate the phase

    for k in range(revolutions_in_series):
        index_of_revolution = int(steps_per_revolution/periode)*k
        # compare neigboring points
        compare_phases = (
            abs(theta[index_of_revolution] - theta[index_of_revolution
                + int(steps_per_revolution/periode)]))
        # compare with 90 degrees
        compare_ana_phases = (
            abs(theta[index_of_revolution] - np.pi/2 - 2*np.pi/100))

        if compare_phases > delta:
            print("pusher is not valid (phases)")
            print("please check the simulation")
            return 1

        if compare_ana_phases > (k+1)*delta:
            print("pusher is not valid (phases_analytical)")
            print("please check the simulation")
            return 2

    print("pusher is valid (phases)")
    return 0


def main(dataPath):
    """ Main function. Calls the functions to compare PIConGPU's simulation
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
    (x_poss, y_poss, x_offSet, y_offSet, x_momentum, y_momentum, charge,
     mass, num_iter, periode) = read_series(series)

    # all radii in the series
    radius = (np.sqrt(x_momentum**2 + y_momentum**2) * params[7] * params[8] /
              (abs(charge[0])*params[5]*B) / params[0] / params[6])
    R_c = radius[0]  # original radius

    # transformation for the correct calculation of the radii by position
    x_poss = x_poss + (R_c * np.sin(2/100 * np.pi))
    y_poss = y_poss - R_c + int(R_c) + (R_c - R_c * np.cos(2/100 * np.pi))

    x_offSet = x_offSet - 32
    y_offSet = y_offSet - 5 - int(R_c)

    # tests
    compare_result_momentum = compare_momentum(x_momentum, y_momentum,
                                               charge, mass, B, params,
                                               num_iter, periode)
    compare_result_positions = compare_positions(x_poss, y_poss,
                                                 x_offSet, y_offSet,
                                                 x_momentum, y_momentum,
                                                 charge, mass, B, params,
                                                 num_iter, periode)
    compare_result_phases = compare_phases(x_poss, y_poss, x_offSet, y_offSet,
                                           x_momentum, y_momentum,
                                           charge, mass, B, params,
                                           num_iter, periode)

    # yield both tests/comparisions a positive result?
    compare_result = (compare_result_momentum + compare_result_positions +
                      compare_result_phases)

    return compare_result


if __name__ == "__main__":
    try:
        arg = sys.argv[1]
    except IndexError:
        raise SystemExit(f"Usage: {sys.argv[0]} <path_to_simulation_data>")
    if len(sys.argv[1:]) > 1:
        raise SystemExit(f"Usage: {sys.argv[0]} <path_to_simulation_data>")
    main(sys.argv[1])
