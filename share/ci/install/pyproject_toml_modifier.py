"""
This file is part of PIConGPU.
Copyright 2023-2024 PIConGPU contributors
Authors: Simeon Ehrig
License: GPLv3+

@file Fix package version in pyproject.toml to a specific version.

Reads an existing pyproject.toml file, sets one or more of the packages to a
fixed version and creates a new pyproject.toml file from the modified
hard set packages and fills up with the remaining packages.

Run `python pyproject_toml_modifier.py --help` to check the usage.
"""

import os
import sys
import argparse
from typing import Dict
import toml
from packaging.requirements import Requirement


def exit_error(text: str):
    """Print error message and exit application with error code 1.

    Parameters
    ----------
        @param text (str): Error message
    """
    # bash annotation to print text with red color
    print(f"\033[0;31mERROR: {text}\033[0m")
    sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "pyproject_toml_modifier",
        description="Reads a existing pyproject.toml file, sets one or more of the packages to a"
        "fixed version and creates a new pyproject.toml file from the modified"
        "hard set packages and fills up with the remaining packages.\n"
        "Versions of the packages are set via environment variables. The"
        "variables need to have the shape of: PYPIC_DEP_VERSION_<package_name>=<version>",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-i",
        type=str,
        required=True,
        help="Set the path of the input pyproject.toml",
    )
    parser.add_argument(
        "-o",
        type=str,
        required=True,
        help="Set the path of the output pyproject.toml",
    )
    parser.add_argument(
        "--ignore_env_args",
        type=str,
        nargs="*",
        default=[],
        help="Ignore these environment variables, which are set to modify the pyproject.toml. "
        "The environment variables starts with `PYPIC_DEP_VERSION_`.",
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

    pyproject_toml = toml.load(args.i)

    # parse dependencies from pyproject.toml
    parsed_dependencies: Dict[str, str] = {}
    for dep in pyproject_toml["project"]["dependencies"]:
        req = Requirement(dep)
        parsed_dependencies[str(req.name)] = str(req.specifier)

    # replace dependency version with versions defined in the environment variables
    for pkg_name, pkg_version in packages.items():
        if pkg_name not in parsed_dependencies:
            exit_error(f"could not find {pkg_name} in pyproject.toml dependencies")
        else:
            parsed_dependencies[pkg_name] = f"=={pkg_version}"

    # replace dependencies in the output pyproject.toml with modified dependency versions
    pyproject_toml["project"]["dependencies"] = []
    for dep_name, dep_version in parsed_dependencies.items():
        pyproject_toml["project"]["dependencies"].append(f"{dep_name}{dep_version}")

    with open(args.o, "w", encoding="utf-8") as output_file:
        toml.dump(pyproject_toml, output_file)
