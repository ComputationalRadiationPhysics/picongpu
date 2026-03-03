"""This file is part of PIConGPU.

Copyright 2026-2026 Edgar Marquardt

Test for the prepareLasyLaser module"""

import argparse
import sys


parser = argparse.ArgumentParser(description="1")


parser.add_argument(
    "-r",
    help="Path to the simulation results",
    dest="path",
    type=str,
)

args = parser.parse_args()

# TODO test, whether the laser field can be found

print("Output validation is not implemented yet")

sys.exit(0)
