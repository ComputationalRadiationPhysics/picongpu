"""
# SPDX-FileCopyrightText: Simeon Ehrig
#
# SPDX-License-Identifier: GPL-3.0-or-later

@file Fix package version in pyproject.toml to a specific version.

Reads an existing pyproject.toml file, sets one or more of the packages to a
fixed version and creates a new pyproject.toml file from the modified
hard set packages and fills up with the remaining packages.

Run `python pyproject_toml_modifier.py --help` to check the usage.
"""

import argparse
import os
import sys

import toml
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


def exit_error(text: str):
    """Print error message and exit application with error code 1.

    Parameters
    ----------
        @param text (str): Error message
    """
    # bash annotation to print text with red color
    print(f"\033[0;31mERROR: {text}\033[0m")
    sys.exit(1)


def updated(requirement_string, packages):
    r = Requirement(requirement_string)
    r.specifier = SpecifierSet(f"=={s}") if (s := packages.get(r.name, False)) else r.specifier
    return str(r)


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
    packages = {
        envvar.split("_")[-1]: os.environ[envvar]
        for envvar in os.environ
        if envvar not in args.ignore_env_args and envvar.startswith("PYPIC_DEP_VERSION_")
    }

    print("Try to set following package to a fix version")
    for pkg_name, pkg_version in packages.items():
        print(f"  {pkg_name} -> {pkg_version}")

    pyproject_toml = toml.load(args.i)

    pyproject_toml["project"]["dependencies"] = [
        updated(d, packages) for d in pyproject_toml["project"]["dependencies"]
    ]

    if not_found := set(packages.keys()).difference(
        map(lambda d: Requirement(d).name, pyproject_toml["project"]["dependencies"])
    ):
        exit_error(f"could not find {not_found=} in pyproject.toml dependencies")

    with open(args.o, "w", encoding="utf-8") as output_file:
        toml.dump(pyproject_toml, output_file)
