"""This file is part of PIConGPU.

Copyright 2026-2026 Edgar Marquardt

Test for the prepareLasyLaser module"""

import argparse
import sys

from lasy.laser import Laser
from lasy.profiles import GaussianProfile

import picongpu.extra.input.prepareLasyLaser as pll

parser = argparse.ArgumentParser(description="1")


parser.add_argument(
    "-r",
    help="Path to the test environment",
    dest="path",
    type=str,
)

args = parser.parse_args()

# adapt the incidentField.param file to current directory
file = open(args.path + "/include/picongpu/param/incidentField.param", "r")
lines = file.readlines()
file.close()

# change the path to the laser file
lines[109] = '                static constexpr char const* filename = "' + args.path + '/diags/test_laser.bp";\n'

file = open(args.path + "/include/picongpu/param/incidentField.param", "w")
file.writelines(lines)
file.close()

# make laser file
w = 2.0e-6 * 50
tau = 2.0564e-16 * 12
try:
    laser = Laser("rt", (0, -2 * tau), (2 * w, 2 * tau), (50, 12), GaussianProfile(7e-7, (1, 0), 1.0, w, tau, 0.0))
    pll.laser_to_openPMD(laser, "test_laser", write_dir=args.path + "/diags", Nx=30, Ny=30)
    sys.exit(0)
except Exception:
    sys.exit(1)
