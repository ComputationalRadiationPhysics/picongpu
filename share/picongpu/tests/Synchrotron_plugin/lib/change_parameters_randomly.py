import os
import numpy as np
import scipy.constants as const
from scipy.integrate import quad
from scipy.special import kv

e = const.elementary_charge  # Elementary charge in C
m_e = const.m_e  # Electron mass in kg
c = const.c  # Speed of light in m/s
hbar = const.hbar  # Reduced Planck constant in J/s

miu0 = const.mu_0  # Vacuum permeability in H/m
eps0 = const.epsilon_0  # Vacuum permittivity in F/m
Es = m_e**2 * c**3 / e / hbar  # Schwinger limit in V/m


def F1(z_q):
    if z_q > 2.9e-6:
        integral = quad(lambda x: kv(5 / 3, x), z_q, np.inf)[0]
        return z_q * integral
    else:
        return 2.15 * z_q ** (1 / 3)


def F2(z_q):
    return z_q * kv(2 / 3, z_q)


def analytical_Propability(delta, gamma, Heff, dt):
    chi = gamma * Heff / Es
    zq = 2 / (3 * chi) * delta / (1 - delta)
    F1_result = F1(zq)
    F2_result = F2(zq)
    numerator = (
        dt * e**2 * m_e * c * np.sqrt(3) * chi * (1 - delta) * (F1_result + 3 * delta * zq * chi / 2 * F2_result)
    )
    denominator = 2 * np.pi * gamma * delta * hbar**2 * eps0 * 4 * np.pi
    return numerator / denominator


# dt is in grid.param as DELTA_T_SI
# Heff is in fieldBackground.param as field_Strength_SI
# gamma is in particle.param as gamma
# all are in ./include/picongpu/param

# params = {
#     "dt": ("grid.param", "DELTA_T_SI", "1e-18"),
#     "Heff": ("fieldBackground.param", "field_Strength_SI", "1e15"),
#     "gamma": ("particle.param", "gamma", "10")
# }

# for key in params:
#     change_param_file(params[key][0], params[key][1], params[key][2])

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


def calculate_dt(gamma, Heff):
    e = 1.602176634e-19  # Elementary charge in Coulombs
    m_e = 9.10938356e-31  # Electron mass in kg
    c = 299792458  # Speed of light in m/s
    hbar = 1.054571817e-34  # Reduced Planck constant in J*s

    miu0 = np.pi * 4.0e-7  # Vacuum permeability in H/m
    eps0 = 1.0 / miu0 / c / c  # Vacuum permittivity in F/m
    Es = 1.3 * 1e18  # Electric field strength in V/m
    chi = gamma * Heff / Es
    numericFactor = e**2 * m_e * c / (hbar**2 * eps0 * 4 * np.pi)
    requirement1 = numericFactor * 1.5 * chi ** (2 / 3) / gamma
    requirement2 = numericFactor * 0.5 * chi ** (1) / gamma
    # dt < 0.1/requirement(1/2)
    return np.min([0.1 / requirement1, 0.1 / requirement2])


def predictNumPhotons(gamma, Heff, dt, Ndt, Nelectrons):
    dt *= Ndt * Nelectrons
    delta = np.logspace(-20, -0.0001, 100)
    P = 0
    for i in range(len(delta) - 1):
        P += quad(lambda delta: analytical_Propability(delta, gamma, Heff, dt), delta[i], delta[i + 1])[0]
    return P


yNum = 0
while yNum < 5e6:
    gamma = [10, 50, 100, 500, 1000]
    Heff = [1e8, 1e10, 1e12, 1e15]

    gamma = gamma[np.random.randint(len(gamma))]
    Heff = Heff[np.random.randint(len(Heff))]
    dt = calculate_dt(gamma, Heff) * 0.95
    dt = dt if dt < 1e-16 else 1e-16
    yNum = predictNumPhotons(gamma, Heff, dt, 4000, 5e5)


changeParams(str(dt), str(Heff), str(gamma))
print("gamma: ", gamma, "Heff: ", Heff, "dt: ", dt)
print(f"Predicted number of photons: {yNum}")

# save parameters in a file in the output directory ./RUN{runNum}
with open("./simOutput/params.txt", "w") as f:
    f.write(f"gamma: {gamma}, Heff: {Heff}, dt: {dt}")
