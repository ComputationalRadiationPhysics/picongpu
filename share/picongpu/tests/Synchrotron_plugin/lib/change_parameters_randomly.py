import os
import numpy as np
from synchrotron_lib import calculate_dt, predictNumPhotons
# dt is in grid.param as DELTA_T_SI
# Heff is in fieldBackground.param as field_Strength_SI
# gamma is in particle.param as gamma
# all are in ./include/picongpu/param

paramsPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../include/picongpu/param")


# read the files change the values and write them back
def change_param_file(file_name, param_name, value):
    with open(os.path.join(paramsPath, file_name), "r") as f:
        lines = f.readlines()
    with open(os.path.join(paramsPath, file_name), "w") as f:
        for line in lines:
            if param_name + " = " in line:
                changedLine = line.split("=")[0] + "= " + value + ";\n"
                # print(changedLine)
                f.write(changedLine)
            else:
                f.write(line)


def changeParams(dt, Heff, gamma):
    change_param_file("grid.param", "DELTA_T_SI", dt)
    change_param_file("fieldBackground.param", "field_Strength_SI", Heff)
    change_param_file("particle.param", "gamma", gamma)


# the code checks if we have enough photons generated for the given parameters:
# gamma, Heff can be any value, dt is calculated based on gamma and Heff
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

# save parameters in a file in the output directory ./RUN{runNum}
with open("./simOutput/params.txt", "w") as f:
    f.write(f"gamma: {gamma}, Heff: {Heff}, dt: {dt}")
