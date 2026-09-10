#!/usr/bin/env python3
#
# Copyright 2026-2026 Edgar Marquardt
#
# This file is part of PIConGPU.
#
# PIConGPU is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PIConGPU is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PIConGPU.
# If not, see <http://www.gnu.org/licenses/>.
#

"""
Test for the Poisson solver. It checks if the divergence of the electric field
matches the charge density.
"""

import argparse
import sys

import openpmd_api as io
from scipy.constants import epsilon_0
import numpy as np

parser = argparse.ArgumentParser(description="1")


parser.add_argument(
    "-r",
    help="Path to the simulation results",
    dest="path",
    type=str,
)

args = parser.parse_args()

# Get the simulation data
filename = args.path + "/openPMD/simData_000000.bp"

series = io.Series(filename, io.Access.read_only)
m = series.iterations[0].meshes

# collect the total charge density
rho_e = m["e_all_chargeDensity"][:, :, :]
rho_i = m["i_all_chargeDensity"][:, :, :]
series.flush()
rho_e *= m["e_all_chargeDensity"].get_attribute("unitSI")
rho_i *= m["i_all_chargeDensity"].get_attribute("unitSI")

rho = rho_e + rho_i
rho = np.transpose(rho)

# get the grid parameters
spacing = np.array(m["e_all_chargeDensity"].get_attribute("gridSpacing"))[::-1] * m[
    "e_all_chargeDensity"
].get_attribute("gridUnitSI")

x = (np.arange(rho.shape[0]) - (rho.shape[0] - 1) / 2) * spacing[0]
y = (np.arange(rho.shape[1]) - (rho.shape[1] - 1) / 2) * spacing[1]
z = (np.arange(rho.shape[2]) - (rho.shape[2] - 1) / 2) * spacing[2]

# collect electric field
E_x = m["E"]["x"][:, :, :]
E_y = m["E"]["y"][:, :, :]
E_z = m["E"]["z"][:, :, :]
series.flush()

E_x *= m["E"]["x"].get_attribute("unitSI")
E_x = np.transpose(E_x)
E_y *= m["E"]["y"].get_attribute("unitSI")
E_y = np.transpose(E_y)
E_z *= m["E"]["z"].get_attribute("unitSI")
E_z = np.transpose(E_z)

# get the divergence of the electric field, which should equal the charge density
diff = np.zeros(E_x.shape)
for ix in range(E_x.shape[0] - 2):
    d0 = (E_x[ix + 1, :, :] - E_x[ix, :, :]) / spacing[0]
    diff[ix + 1, :, :] += d0

for iy in range(E_y.shape[1] - 2):
    d0 = (E_y[:, iy + 1, :] - E_y[:, iy, :]) / spacing[1]
    diff[:, iy + 1, :] += d0

for iz in range(E_z.shape[2] - 2):
    d0 = (E_z[:, :, iz + 1] - E_z[:, :, iz]) / spacing[2]
    diff[:, :, iz + 1] += d0

diff *= epsilon_0

# comparing the two
rerr = np.abs((diff[1:-1, 1:-1, 1:-1] - rho[1:-1, 1:-1, 1:-1]) / diff[1:-1, 1:-1, 1:-1])

print("maximum relative error:", np.max(rerr))
print(
    "maximum relative error for absolute values above 1e-3 of the maximum charge density:",
    np.max(rerr[np.abs(diff[1:-1, 1:-1, 1:-1]) > 1e-3 * np.max(np.abs(diff))]),
)

if np.max(rerr[np.abs(diff[1:-1, 1:-1, 1:-1]) > 1e-3 * np.max(np.abs(diff))]) > 0.005:
    print("relative error too high!")
    sys.exit(1)

if np.max(rerr) > 0.9:
    print("relative error too high!")
    sys.exit(1)

sys.exit(0)
