# /* Copyright 2014-2024 Filip Optolowicz
#  *
#  * This file is part of PIConGPU.
#  *
#  * PIConGPU is free software: you can redistribute it and/or modify
#  * it under the terms of the GNU General Public License as published by
#  * the Free Software Foundation, either version 3 of the License, or
#  * (at your option) any later version.
#  *
#  * PIConGPU is distributed in the hope that it will be useful,
#  * but WITHOUT ANY WARRANTY; without even the implied warranty of
#  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  * GNU General Public License for more details.
#  *
#  * You should have received a copy of the GNU General Public License
#  * along with PIConGPU.
#  * If not, see <http://www.gnu.org/licenses/>.
#  */

import numpy as np
import scipy.constants as const

# Constants
e = const.elementary_charge  # Elementary charge in C
m_e = const.m_e  # Electron mass in kg
c = const.c  # Speed of light in m/s
hbar = const.hbar  # Reduced Planck constant in J/s
miu0 = const.mu_0  # Vacuum permeability in H/m
eps0 = const.epsilon_0  # Vacuum permittivity in F/m
Es = m_e**2 * c**3 / e / hbar  # Schwinger limit in V/m


#  Function to calculate Heff as described in the paper:
#  "Extended particle-in-cell schemes for physics in ultrastrong laser fields: Review and
#  developments" by A. Gonoskov et.Al.
def Heff_(v, B, E):
    vmag = (v**2).sum()
    return np.sqrt(((np.cross(v, B) + E) ** 2).sum(-1) - (np.dot(v / vmag, E)) ** 2)


# calculate Heff given a0 assuming 800nm laser
def Heff_a0(a0, gamma):
    w = 2 * np.pi * c / 800e-9  # Angular frequency for 800nm laser
    El = a0 * m_e * w * c / e  # Electric field strength in V/m
    # print("El", El)
    beta = np.sqrt(1 - 1 / gamma**2)  # Beta value for the particles
    vel = c * beta

    magnitude = El
    E = np.zeros(3)
    E[0] = magnitude
    B = np.zeros(3)
    B[2] = -magnitude / c

    VecVel = np.array([0, vel, 0])
    return Heff_(VecVel, B, E)


def calculate_dt(gamma, Heff):
    chi = gamma * Heff / Es
    numericFactor = e**2 * m_e * c / (hbar**2 * eps0 * 4 * np.pi)
    requirement1 = numericFactor * 1.5 * chi ** (2 / 3) / gamma
    requirement2 = numericFactor * 0.5 * chi ** (1) / gamma
    # dt < 0.1/requirement(1/2)
    return np.min([0.1 / requirement1, 0.1 / requirement2])


def main():
    print("Synchrotron Radiation Requirements Checker")
    print("----------------------------------------")

    choice = input("Do you want to specify (1) Heff, (2) B and E, or (3) a0 and gamma? ")

    if choice == "1":
        Heff_estimate = float(input("Enter an estimate of the maximal Heff value (in Tesla): "))
        gamma_estimate = float(input("Enter an estimate of the maximal gamma value: "))
    elif choice == "2":
        B_estimate = float(input("Enter an estimate of the maximal B value (in Tesla): "))
        E_estimate = float(input("Enter an estimate of the maximal E value (in V/m): "))
        gamma_estimate = float(input("Enter an estimate of the maximal gamma value: "))
        beta = np.sqrt(1 - 1 / gamma_estimate**2)  # Beta value for the particles
        vel = c * beta
        Heff_estimate = Heff_(np.array([0, vel, 0]), np.array([0, 0, B_estimate]), np.array([E_estimate, 0, 0]))
    elif choice == "3":
        a0_estimate = float(input("Enter an estimate of the maximal a0 value: "))
        gamma_estimate = float(input("Enter an estimate of the maximal gamma value: "))
        Heff_estimate = Heff_a0(a0_estimate, gamma_estimate)
    else:
        print("Invalid choice. Exiting.")
        return

    dt = calculate_dt(gamma_estimate, Heff_estimate)

    print(f"Based on the estimates, the recommended dt value is: {dt:.2e} s")


if __name__ == "__main__":
    main()
