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

JtoeV = e**-1  # Conversion factor from Joules to eV


def F1(z_q):
    if z_q > 2.9e-6:  # below this value the integral is not accurate and we use the approximation for z_q << 1
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


def calculate_dt(gamma, Heff):
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


def electron_energy(gamma):
    return gamma * m_e * c**2 * JtoeV  # Energy of the electrons in eV


# Returns energy in eV without the rest mass energy
def momentum_to_energy(momentum, mass):
    return (np.sqrt((momentum * const.c) ** 2 + (mass * const.c**2) ** 2) - mass * const.c**2) * JtoeV
