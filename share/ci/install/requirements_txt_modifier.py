import os
import sys

"""
This file is part of the PIConGPU.
Copyright 2023 PIConGPU contributors
Authors: Simeon Ehrig
License: GPLv3+
"""

"""@file Fix package version in requirement.txt to a specific version.

Reads an existing requirements.txt file, sets one or more of the packages to a
fixed version and creates a new requirements.txt file from the modified
hard set packages and fills up with the remaining packages.

@param First application argument: Path to the original requirements.txt
@param Second application argument: Path to the mew requirements.txt

@attention Versions of the packages are set via environment variables. The
variables need to have the shape of: PYPIC_DEP_VERSION_<package_name>=<version>
"""


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
    if len(sys.argv) < 3:
        print(
            "Set path to requirements.txt as first argument and output path as"
            " second argument."
        )
        exit(1)

    # parse environment variables
    packages = {}
    for envvar in os.environ:
        if envvar.startswith("PYPIC_DEP_VERSION_"):
            packages[envvar.split("_")[-1]] = os.environ[envvar]

    print("Try to set following package to a fix version")
    for pkg_name, pkg_version in packages.items():
        print(f"  {pkg_name} -> {pkg_version}")

    with open(sys.argv[1], "r", encoding="utf-8") as input:
        with open(sys.argv[2], "w", encoding="utf-8") as output:
            for line in input.readlines():
                input_pkg_name = line.split(" ")[0]
                if input_pkg_name in packages:
                    output.write(
                        f"{input_pkg_name} == {packages[input_pkg_name]}\n"
                    )
                    packages.pop(input_pkg_name)
                else:
                    output.write(line)

    # debug messages
    for pkg_name in packages.keys():
        print(
            cs(
                f"WARNING: could not find {pkg_name} in requirements.txt",
                "Yellow",
            )
        )
