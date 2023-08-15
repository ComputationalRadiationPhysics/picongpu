"""
This file is part of PIConGPU.

Copyright 2023 PIConGPU contributors
Authors: Hannes Wolf
License: GPLv3+

Test script to determine if a Boris pusher works correctly. This is done by
calculating the phase shift caused by the pusher.

The phase shift is proportional to (dt)**2. The script checks whether the
calculated exponent of (dt)**x is in the interval [1.9, 2.1] or not.
The standard deviation should be < 0.05.

"""

import openpmd_api as io
import numpy as np
import sys


class compare:
    def __init__(self, path, B):
        self.path = path
        self.B = B

    def get_params(self):
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
            unit_time (float): Conversion factor of arbitrary PIConGPU time
                               units to SI
            unit_charge (float): Conversion factor of arbitrary PIConGPU charge
                                 units to SI
            unit_length (float): Conversion factor of arbitrary PIConGPU length
                                 units to SI

        """
        # load series
        series = io.Series(self.path, io.Access.read_only)
        indices = np.array(series.iterations)
        i = series.iterations[indices[0]]

        if len(series.iterations) == 1:
            raise ValueError("There is just 1 iteration in the series"
                             "make sure, there are at least two")

        elif len(np.array(i.particles)) != 1:
            raise ValueError("There is not only 1 particle in the series"
                             "make sure, there is onle least one")

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

            params = [
                cell_depth, cell_height, cell_width, dt, unit_time,
                unit_charge, unit_length, unit_mass, unit_speed, unit_bfield
                ]

            self.series = series
            self.params = params

    def read_series(self):
        """Reads the position and momentum of the particle in the x-y-plane
        provided by the simulation, as well as the charge and mass of the
        particle.

        Parameters:
        series: simulation data

        Returns: (self.variable)
        x/y_poss (np.array): Relative in-Cell positions of the particle in the
                             series
        x/y_offSet (np.array): All node-positions of the particle in the series
        periode (int): openPMD periode of the simulation
        charge (float): Charge of particle
        mass (float): Mass of particle
        x/y_momentum (float): All momenta of the particle in the series
        R_c (float): Radius of gyration of the particle at timestep 0

        """

        num_iter = len(self.series.iterations)

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

        i = self.series.iterations[0]
        # charge
        self.charge = (i.particles["e"]["charge"]
                       [io.Mesh_Record_Component.SCALAR])
        # mass
        self.mass = i.particles["e"]["mass"][io.Mesh_Record_Component.SCALAR]

        # calculating the openPMD-periode for later use
        indices = np.array(self.series.iterations)
        periode = indices[1] - indices[0]  # openPMD-periode

        for iteration in indices:
            i = self.series.iterations[iteration]  # current position

            # node coordinates
            pos2_x = i.particles["e"]["position"]["x"].load_chunk()
            pos2_y = i.particles["e"]["position"]["y"].load_chunk()

            # InCell coordinates
            pos_off2_x = i.particles["e"]["positionOffset"]["x"].load_chunk()
            pos_off2_y = i.particles["e"]["positionOffset"]["y"].load_chunk()

            p_x = i.particles["e"]["momentum"]["x"].load_chunk()
            p_y = i.particles["e"]["momentum"]["y"].load_chunk()

            self.series.flush()

            array_index = int(iteration/periode)
            # save the read data in arrays for later usage
            x_poss[array_index] = pos2_x
            y_poss[array_index] = pos2_y

            x_offSet[array_index] = pos_off2_x
            y_offSet[array_index] = pos_off2_y

            x_momentum[array_index] = p_x
            y_momentum[array_index] = p_y

        # all radii in the series
        radius = (np.sqrt(x_momentum**2 + y_momentum**2) * self.params[7]
                  * self.params[8] /
                  (abs(self.charge[0]) * self.params[5] * self.B) /
                  self.params[0] / self.params[6])

        R_c = radius[0]  # original radius

        # half_step
        steps_per_turn = 0.825e-12 / (self.params[3] * self.params[4])
        halfStepPhase = np.pi/steps_per_turn

        # coordinate transformation to reposition the origin at the center of
        # the circle
        x_poss = x_poss + (R_c * np.sin(halfStepPhase))
        y_poss = y_poss - R_c + int(R_c) + (R_c - R_c * np.cos(halfStepPhase))

        x_offSet = x_offSet - 32
        y_offSet = y_offSet - 5 - int(R_c)

        self.periode = periode

        # in-cell position
        self.x_poss = x_poss
        self.y_poss = y_poss

        # cells
        self.x_offSet = x_offSet
        self.y_offSet = y_offSet

        # momenta of the series
        self.x_momentum = x_momentum
        self.y_momentum = y_momentum

        self.R_c = R_c

    def get_total_phasediff(self):
        """Calculates the total phase difference accumulated during the
        simulation.

        Returns:
        total_phasediff (float): Total phase difference
        timestep (int): Timestep used in the series

        """
        x = self.x_poss + self.x_offSet     # x coordinates
        y = self.y_poss + self.y_offSet     # y coordinates

        res_cos = x/np.sqrt(x**2+y**2)
        theta = np.arccos(res_cos)

        timestep = self.params[3] * self.params[4]

        total_phasediff = 0
        for counter in range(0, len(theta)-1, 1):
            total_phasediff += abs(theta[counter+1] - theta[counter])

        return total_phasediff, timestep


def main():
    """ Main function. Creates a compare object and calls its functions to
        compare PIConGPU's phase difference scaling with the expectation.

        Returns:
        compare_result (int): 0 if the simulated scaling can be regarded as
                              equivalent to the expectation;
                              Non-zero if they differ.

    """
    B = 50
    # array for total phase differences
    phaseDifferences = np.zeros(len(sys.argv)-1)
    timesteps = np.zeros(len(sys.argv)-1)

    for j in range(1, len(sys.argv)):
        path = sys.argv[j]
        phase_class = compare(path, B)
        phase_class.get_params()

        # extract all relevant data
        phase_class.read_series()

        phaseDifferences[j-1], timesteps[j-1] = (
                                            phase_class.get_total_phasediff())

    quotients = np.zeros(len(phaseDifferences)-1)
    for j in range(len(phaseDifferences)-1):
        quotients[j] = phaseDifferences[j+1]/phaseDifferences[j]

    b = np.mean(quotients)  # base b
    expo = np.log(b)/np.log(2)  # exponent x of b = 2**x
    sigma_b = (max(quotients) - min(quotients))/2  # std of b
    sigma_x = 1/(b*np.log(2))*sigma_b

    # expected value for scaling of the total phase difference of the
    # Boris Pusher
    # Reference: Ripperda et al. A Comprehensive Comparison of Relativistic
    # Particle Integrators  https://doi.org/10.3847/1538-4365/aab114
    expect_x = 2
    # expected standard deviation = 0

    # maximal deviation from the expected values
    epsilon = 0.1
    delta = 0.05
    # These deviation values are arbitrary, yet reasonable.
    # They are chosen to be double the deviation found in recent PIConGPU runs
    # (2023-09-07)

    if abs(expo - expect_x) > epsilon:
        x_compare = 45
        print("pusher is NOT valid (exponent)")
    else:
        x_compare = 0
        print("pusher is valid (exponent)")

    if abs(sigma_x) > delta:
        sigma_compare = 44
        print("pusher is NOT valid (standard-deviation")
    else:
        sigma_compare = 0
        print("pusher is valid (standard-deviation)")

    compare_result = x_compare + sigma_compare

    return compare_result


if __name__ == "__main__":
    main()
