#!/usr/bin/env python
"""
This file is part of PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Pawel Ordyna
License: GPLv3+
"""

import argparse
import os
from ThermalizationVerifier import ThermalizationVerifier

smilei_import_error = None
try:
    from SmileiThermalization import SmileiThermalization
except ImportError as import_error:
    smilei_import_error = import_error

    class SmileiThermalization:
        def __init__(self, *args, **kwargs):
            raise Exception(
                "Smilei data import class could not be imported. "
                "Check module requirements or run without the "
                "smilei_dir option."
            ) from smilei_import_error


def main():
    parser = argparse.ArgumentParser(
        description="It calculates electron and "
        "ion temperatures for all "
        "simulation steps and plots "
        "them together with a "
        "theretical curve."
    )
    parser.add_argument(
        "dir",
        nargs="?",
        help="simulation directory containing the simOutput " "directory",
        default=os.getcwd(),
    )
    parser.add_argument(
        "--coulomb_log",
        help="Coulomb logarithm for theoretical calculation. "
        "If not set it uses "
        "dynamic caluclation based on a theretical "
        "model for electron ion collisions",
        type=float,
    )
    parser.add_argument(
        "--n_cells",
        help="number of cells to use to " "caluculate average values, " "by default all availible",
        type=int,
    )
    parser.add_argument("--file", help="figure file name", type=str)
    parser.add_argument(
        "--smilei_dir",
        help="path to smilei test simulation if" " it should be plotted together" " with PIConGPU data",
        type=str,
    )
    parser.add_argument("--file_debug", help="debug values figure file name", type=str)
    parser.add_argument(
        "--disable_main",
        dest="plot_main",
        action="store_false",
        help="disables plotting main plot",
    )
    parser.set_defaults(plot_main=True)
    parser.add_argument(
        "--disable_debug",
        dest="plot_debug",
        action="store_false",
        help="disables plotting debug values plot",
    )
    parser.set_defaults(plot_debug=True)
    args = parser.parse_args()

    verifier = ThermalizationVerifier(args.dir)
    verifier.calculate_temperatures(n_cells=args.n_cells)
    verifier.calculate_theretical_values(args.coulomb_log)
    if args.smilei_dir is not None:
        smilei_sim = SmileiThermalization(args.smilei_dir)
    else:
        smilei_sim = None
    if args.plot_main:
        verifier.plot(to_file=True, file_name=args.file, smilei_sim=smilei_sim)
    if args.plot_debug:
        verifier.plot_debug_values(to_file=True, file_name=args.file_debug, smilei_loader=smilei_sim)


if __name__ == "__main__":
    main()
