import requests
import sys
import yaml
import re
from typing import List, Dict, Callable
import pkg_resources
import copy
import typeguard

"""
This file is part of PIConGPU.
Copyright 2023-2023 PIConGPU contributors
Authors: Simeon Ehrig
License: GPLv3+
"""

"""@file Generate different CI test jobs for different Python version and
         depending on a requirement.txt.

Prints yaml code for a GitLab CI child pipeline to stdout. The test parameters
are split in two kinds of inputs. The Python versions to test are defined in
the script, also the names of dependencies to test and it's test strategy. The
version range of the dependencies to test are defined in the passed
requirements.txt files. The paths of the requirements.txt are set via the
application arguments.

First, the script reads the requirements.txt files. If a dependency is marked
as to be tested in the script, it calculates the test versions.
For this it downloads all available versions from pypi.org for each package.
Afterwards it filters the versions via a filter strategy. For example, take
all release versions or take each latest major version. Than the script
removes all versions, which are not supported, as defined in the combined
requirements.txt.
The result a complete list of all Python- and dependency- versions to test.

In the second part, the script creates the full combination matrix for all test
versions and creates a CI job for each combination. Each job is printed to
stdout.

The number of combinations depends on:
- number of supported Python version
- number of dependencies to test
- the test strategy of each dependencies to test
- versions restrictions in the requirements.txt
- releases of the dependencies

@param First application argument: Path to the requirements.txt
"""


# caches the parsed dependencies of the requirements.txt here
req_versions: Dict[str, pkg_resources.Requirement] = {}


def cs(text: str, color: str) -> str:
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


def get_all_pypi_versions(package_name: str) -> List[str]:
    """Returns all release versions of a package registered on pypi.org

    Parameters
    ----------
        @param package_name (str): Name of the searched package.

    Returns
    -------
        @return List[str]: List of release versions.
    """

    url = f"https://pypi.org/pypi/{package_name}/json"

    res = requests.get(url, timeout=5)

    data = res.json()
    # remove all release candidates, alpha and beta releases
    # allows only version strings containing numbers and dots
    versions = [v for v in data["releases"] if re.match(r"^[0-9\.]*$", v)]

    return sorted(versions, key=pkg_resources.parse_version, reverse=True)


def get_all_major_pypi_versions(package_name):
    """Returns the latest release versions of each major release of a package
    registered on pypi.org

    Parameters
    ----------
        @param package_name (str): Name of the searched package.

    Returns
    -------
        @return List[str]: List of release versions.
    """
    all_versions = get_all_pypi_versions(package_name)
    version_map = {}

    for version in all_versions:
        parsed_version = pkg_resources.parse_version(version)
        # all versions are sorted from the highest to the lowest
        # therefore no complex comparison of the version is required
        # simply take the first appearance of a major version
        if parsed_version.major not in version_map:
            version_map[parsed_version.major] = parsed_version

    return [str(v) for v in version_map.values()]


@typeguard.typechecked
def combine_requirements(one: pkg_resources.Requirement, two: pkg_resources.Requirement) -> pkg_resources.Requirement:
    """combine two requirements for one dependency"""
    if one.project_name != two.project_name:
        print(f"requirements for {one.project_name} and {two.project_name} are incomparable!")
        exit(1)

    # accumulate all specs
    specs: list = copy.deepcopy(one.specs)
    specs.extend(two.specs)

    if len(specs) != 0:
        last_tuple = specs.pop()
        #                                             operator        version
        # example:           typeguard                ">="            "4.2.11"
        requirement_string = one.project_name + " " + last_tuple[0] + last_tuple[1]
        for entry in specs:
            # further version restrictions, e.g. ", <=" + "4.3.0"
            requirement_string += "," + entry[0] + entry[1]
    else:
        requirement_string = one.project_name

    return pkg_resources.Requirement(requirement_string)


@typeguard.typechecked
def read_in_requirement_files(requirement_file_names: list[str]) -> dict[str, pkg_resources.Requirement]:
    """read in requirements from list of requirement files"""
    total_requirements = {}
    for requirement_file_name in requirement_file_names:
        with open(requirement_file_name, "r", encoding="utf-8") as requirement_file:
            try:
                # read in file
                parsed_requirements = pkg_resources.parse_requirements(requirement_file)

                # accumulate
                for requirement in parsed_requirements:
                    if requirement.project_name not in total_requirements:
                        # does not exist yet
                        total_requirements[requirement.project_name] = requirement
                    else:
                        # already exists
                        total_requirements[requirement.project_name] = combine_requirements(
                            requirement, total_requirements[requirement.project_name]
                        )

            except Exception:
                # ignore all lines, which cannot be parsed
                # e.g. `-r extra/requirements.txt`
                pass
    return total_requirements


def get_supported_versions(package_name: str, versions: List[str]) -> List[str]:
    """Take a list of package versions all removes all version, which are not
    supported by the requirements.txt.

    Parameters
    ----------
        @param package_name (str): Name of the package.
        @param versions (List[str]): List to be filtered

    Returns
    -------
        @return List[str]: filtered list
    """
    # use global variable to cache parsed versions from the requirements.txt
    global req_versions
    filtered_versions = []

    requirement_file_names = sys.argv[1:]

    if not req_versions:
        req_versions = read_in_requirement_files(requirement_file_names)

    if package_name not in req_versions:
        print(cs(f"ERROR: {package_name} is not defined in {sys.argv[1:]}", "Red"))
        exit(1)

    for v in versions:
        if str(v) in req_versions[package_name]:
            filtered_versions.append(v)

    return filtered_versions


class Job:
    """The Job class stores a single GitLab CI job description.
    It actual replace the dictionary data structure {job_name : { # job_body }}
    and gives the guaranty, that there is only one key on the dict top level,
    which makes it much easier to access the job name.
    """

    def __init__(self, name: str, body: Dict):
        """Creates a Job object, see class description.

        Parameters
        ----------
            @param name (str): Name of the job
            @param body (Dict): Body of the job. Contains for example the
                entries `variables`, `script` and so one.
        """
        self.name = name
        self.body = body

    def yaml_dumps(self) -> str:
        """Generate yaml representation of the job.

        Returns
        -------
            @return str: Yaml representation as string.
        """
        return yaml.dump({self.name: self.body})


def extend_job_with_test_requirement(job: Job, package_name: str, package_version: str) -> Job:
    """Copies the input job, adds a new variable to the variables section of
    the copied job and return it.

    Parameters
    ----------
        @param job (Job): Job to be extent
        @param package_name (str): Name of the package to add
        @param package_version (str): Version of the package to add

    Returns
    -------
        @return Job: Copy of the input job, extend in the variable section a
        variable containing package name and version.
    """
    job_copy_name = job.name + "_" + package_name + package_version
    job_copy = Job(job_copy_name, job.body)
    job_copy.body["variables"]["PYPIC_DEP_VERSION_" + package_name] = package_version

    return job_copy


def construct_job(
    job: Job,
    current_test_pkgs: List[str],
    test_pkg_versions: Dict[str, List[str]],
):
    """Recursive function to construct all test jobs.

    Starts with an initial job, passed via the argument job. The initial jobs
    contains attributes like `image`, `extends`, `variables` and so one. Each
    function call adds a variable to the `variables` section, which describes
    which version of a dependency should be tested.

    The "counting variable" is the length of the current_test_pkg. Each
    function call the function takes the first element and adds a variable to
    the job depending on the package name. Then it calls the function again and
    remove the first argument. If only one argument is left, the functions adds
    the variable, generates the job yaml and prints to stdout.

    Parameters
    ----------
        @param job (Job): Current job to extent.
        @param current_test_pkgs (List[str]): Current package to add.
        @param test_pkg_versions (Dict[str, List[str]]): Versions of each
            package.
    """
    package_name = current_test_pkgs[0]

    if len(current_test_pkgs) == 1:
        for package_version in test_pkg_versions[package_name]:
            extended_job = extend_job_with_test_requirement(job, package_name, package_version)
            print(extended_job.yaml_dumps())
    else:
        for package_version in test_pkg_versions[package_name]:
            construct_job(
                extend_job_with_test_requirement(job, package_name, package_version),
                current_test_pkgs[1:],
                test_pkg_versions,
            )


def print_job_yaml(test_pkg_versions: Dict[str, List[str]]):
    """Prints all GitLab CI jobs on stdout.

    Parameters
    ----------
        @param test_pkg_versions (Dict[str, List[str]]): Dependency versions to
            test.
    """
    # contains the .base_pypicongpu_quick_test base job
    print(yaml.dump({"include": "/share/ci/pypicongpu.yml"}))

    for pyVer in PYTHON_VERSIONS:
        job = Job(
            name="PyPIConGPU_Python" + pyVer,
            body={
                "variables": {
                    "PYTHON_VERSION": pyVer + ".*",
                    "CI_CONTAINER_NAME": "ubuntu20.04",
                },
                "extends": ".base_pypicongpu_quick_test",
            },
        )
        construct_job(job, list(test_pkg_versions.keys()), test_pkg_versions)


# Python versions to test
PYTHON_VERSIONS: List[str] = ["3.10", "3.11", "3.12"]
# Define, which dependencies should be explicit tests.
# The key is the name of the package, and function returns the versions to
# test.
# If a package is not define in the list, but defined in the requirements.txt,
# pip decides which version is used.
PACKAGES_TO_TEST: Dict[str, Callable] = {
    "typeguard": get_all_major_pypi_versions,
    "jsonschema": get_all_major_pypi_versions,
    "picmistandard": get_all_pypi_versions,
    "pydantic": get_all_major_pypi_versions,
    "referencing": get_all_major_pypi_versions,
}

if __name__ == "__main__":
    # note, script name is sys.argv[0] rest are bash inputs
    if len(sys.argv) < 2:
        print(cs("ERROR: pass path(s) to one or more requirements.txt as arguments", "Red"))
        exit(1)

    test_pkg_versions: Dict[str, List[str]] = {}

    for pkg, version_func in PACKAGES_TO_TEST.items():
        test_pkg_versions[pkg] = get_supported_versions(pkg, version_func(pkg))

    print_job_yaml(test_pkg_versions)
