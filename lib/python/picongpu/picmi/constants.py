"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from scipy import constants as consts

c = consts.c
ep0 = consts.epsilon_0
mu0 = consts.mu_0
q_e = consts.elementary_charge
m_e = consts.electron_mass
m_p = consts.proton_mass

# exact energy equivalents in SI (joule), derived from the exact elementary charge
eV = consts.electron_volt
keV = 1e3 * consts.electron_volt

# byte-size multipliers, so memory values can be written as `350 * MiB` etc.
# (SI decimal prefixes are powers of 1000; IEC binary prefixes are powers of 1024)
B = 1
kB = 1_000
MB = 1_000_000
GB = 1_000_000_000
KiB = 1024
MiB = 1024**2
GiB = 1024**3
