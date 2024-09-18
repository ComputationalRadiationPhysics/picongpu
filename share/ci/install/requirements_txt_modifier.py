"""
This file is part of PIConGPU.
Copyright 2023-2023 PIConGPU contributors
Authors: Simeon Ehrig
License: GPLv3+

@file Fix package version in requirement.txt to a specific version.

Reads an existing requirements.txt file, sets one or more of the packages to a
fixed version and creates a new requirements.txt file from the modified
hard set packages and fills up with the remaining packages.

Run `python requirements_txt_modifier.py --help` to check the usage.
"""

import os
import sys
import argparse


def cs(text: str, color: str):
    """Print the text in a different color on the command line. The text after
       the function has the default color of the command line.

    Parameters
    ----------
        @param text (str): text to be colored
        @param color (str): Name of the color. If wrong color or empty, use
            default color of the command line.

    Returns
    -------
        @return str: text with bash pre and post string for coloring
    """

    if color is None:
        return text

    output = ""
    if color == "Red":
        output += "\033[0;31m"
    elif color == "Green":
        output += "\033[0;32m"
    elif color == "Yellow":
        output += "\033[1;33m"

    return output + text + "\033[0m"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "requirements_txt_modifier",
        description="Reads an existing requirements.txt file, sets one or more of the packages to a"
        "fixed version and creates a new requirements.txt file from the modified"
        "hard set packages and fills up with the remaining packages.\n"
        "Versions of the packages are set via environment variables. The"
        "variables need to have the shape of: PYPIC_DEP_VERSION_<package_name>=<version>",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="Set the path of the input requirements.txt",
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Set the path of the output requirements.txt",
    )
    parser.add_argument(
        "--ignore_env_args",
        type=str,
        nargs="*",
        default=[],
        help="Ignore these environment variables, which are set to modify the requirements.txt. The environment variables starts with `PYPIC_DEP_VERSION_`.",
    )
    args = parser.parse_args()

    # parse environment variables
    packages = {}
    for envvar in os.environ:
        if envvar not in args.ignore_env_args:
            if envvar.startswith("PYPIC_DEP_VERSION_"):
                packages[envvar.split("_")[-1]] = os.environ[envvar]

    print("Try to set following package to a fix version")
    for pkg_name, pkg_version in packages.items():
        print(f"  {pkg_name} -> {pkg_version}")

    with open(args.i, "r", encoding="utf-8") as input_file:
        with open(args.o, "w", encoding="utf-8") as output_file:
            for line in input_file.readlines():
                input_pkg_name = line.split(" ")[0]
                if input_pkg_name in packages:
                    output_file.write(f"{input_pkg_name} == {packages[input_pkg_name]}\n")
                    packages.pop(input_pkg_name)
                else:
                    output_file.write(line)

    # debug messages
    for pkg_name in packages.keys():
        print(
            cs(
                f"ERROR: could not find {pkg_name} in requirements.txt",
                "Red",
            )
        )
        sys.exit(1)
