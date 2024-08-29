"""
Copyright 2024 Filip Optolowicz

This file is part of PIConGPU.

PIConGPU is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PIConGPU is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with PIConGPU.
If not, see <http://www.gnu.org/licenses/>.
"""

# Usage:
# This script changes:
#  - DELTA_T_SI         in file simulation.param
#  - field_Strength_SI  in file fieldBackground.param
#  - gamma              in file particle.param
# gamma and Heff are randomly chosen from the lists gamma and Heff
# and saves the configuration in the file ./simOutput/params.txt

# This script is run from the ../bin/ci.sh

# This script assumes that the files:
# grid.param, fieldBackground.param, particle.param
# are in the directory ../include/picongpu/param
# and that the output directory ./simOutput exists


import numpy as np
from synchrotron_lib import calculate_dt, predictNumPhotons
from pathlib import Path

paramsPath = Path(__file__).absolute().parent / "../include/picongpu/param"


# read the files change the values and write them back
def change_param_file(file_name, param_name, value):
    filePath = paramsPath / file_name
    with open(filePath, "r") as f:
        lines = f.readlines()
    with open(filePath, "w") as f:
        for line in lines:
            if param_name + " = " in line:
                changedLine = line.split("=")[0] + "= " + value + ";\n"
                # print(changedLine)
                f.write(changedLine)
            else:
                f.write(line)


def changeParams(dt, Heff, gamma):
    change_param_file("simulation.param", "DELTA_T_SI", dt)
    change_param_file("fieldBackground.param", "field_Strength_SI", Heff)
    change_param_file("particle.param", "gamma", gamma)


# the code checks if we have enough photons generated for the given parameters:
# gamma, Heff can be any value, dt is calculated based on gamma and Heff
def main():
    yNum = 0
    while yNum < 5e6:
        gamma = [10, 50, 100, 500, 1000]  # include all gamma values you want to check
        Heff = [1e13, 1e14, 1e15, 1e16, 1e17, 1e18]  # similarly with Heff

        gamma = gamma[np.random.randint(len(gamma))]
        Heff = Heff[np.random.randint(len(Heff))]
        dt = calculate_dt(gamma, Heff) * 0.95
        dt = dt if dt < 1e-16 else 1e-16  # if dt is larger than the grid condition, set it to the grid condition
        yNum = predictNumPhotons(gamma, Heff, dt, 4000, 5e5)

    # make Heff, gamma and dt strings in scientific notation
    Heff = "{:.2e}".format(Heff)
    gamma = "{:.3e}".format(gamma)
    dt = "{:.2e}".format(dt)
    changeParams(dt, Heff, gamma)
    print("gamma: ", gamma, "Heff: ", Heff, "dt: ", dt)
    print(f"Predicted number of photons: {yNum}")

    # save parameters in a file in the output directory ./simOutput
    with open("./simOutput/params.txt", "w") as f:
        f.write(f"gamma: {gamma}, Heff: {Heff}, dt: {dt}")


if __name__ == "__main__":
    main()
