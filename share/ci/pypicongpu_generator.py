from typing import List, Dict, Callable
import sys
import tomllib
import re
import packaging.requirements
import packaging.version
import requests
import yaml
import typeguard

"""
This file is part of PIConGPU.
Copyright 2023-2024 PIConGPU contributors
Authors: Simeon Ehrig
License: GPLv3+
"""

"""@file Generate different CI test jobs for different Python version and
         depending on a pyproject.toml.

Prints yaml code for a GitLab CI child pipeline to stdout. The test parameters
are split in two kinds of inputs. The Python versions to test are defined in
the script, also the names of dependencies to test and it's test strategy. The
version range of the dependencies to test are defined in the passed
pyproject.toml files. The path of the pyproject.toml is set via the
application arguments.

First, the script reads the pyproject.toml files. If a dependency is marked
as to be tested in the script, it calculates the test versions.
For this it downloads all available versions from pypi.org for each package.
Afterwards it filters the versions via a filter strategy. For example, take
all release versions or take each latest major version. Than the script
removes all versions, which are not supported, as defined in the pyproject.toml.
The result a complete list of all Python- and dependency- versions to test.

In the second part, the script creates the full combination matrix for all test
versions and creates a CI job for each combination. Each job is printed to
stdout.

The number of combinations depends on:
- number of supported Python version
- number of dependencies to test
- the test strategy of each dependencies to test
- versions restrictions in the pyproject.toml
- releases of the dependencies

@param First application argument: Path to the pyproject.toml
"""


@typeguard.typechecked
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


@typeguard.typechecked
def exit_error(text: str):
    """Print error message and exit application with error code 1.

    Parameters
    ----------
        @param text (str): Error message
    """
    print(cs(f"ERROR: {text}", "Red"))
    sys.exit(1)


@typeguard.typechecked
def get_all_pypy_releases(package_name: str, version_strategy: Callable[[str], List[str]]) -> List[str]:
    """Return release version of given package depending on the version_strategy.

    Parameters
    ----------
        @param package_name (str): Name of the package
        @param version_strategy (Callable[[str], List[str]]): The version strategy decides which
            release version are returned

    Returns
    -------
        @return List[str]: List of versions
    """
    return version_strategy(package_name)


@typeguard.typechecked
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

    return sorted(versions, key=packaging.version.parse, reverse=True)


@typeguard.typechecked
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
        parsed_version = packaging.version.parse(version)
        # all versions are sorted from the highest to the lowest
        # therefore no complex comparison of the version is required
        # simply take the first appearance of a major version
        if parsed_version.major not in version_map:
            version_map[parsed_version.major] = parsed_version

    return [str(v) for v in version_map.values()]


@typeguard.typechecked
def get_supported_versions(package_name: str, versions: List[str], pyproject_toml: Dict) -> List[str]:
    """Take a list of package versions and remove all versions, which are not supported by the
    pyproject.toml.

    Parameters
    ----------
        @param package_name (str): Name of the package.
        @param versions (List[str]): List to be filtered
        @param pyproject_toml (Dict): The pyproject.toml

    Returns
    -------
        @return List[str]: filtered list
    """
    for dep in pyproject_toml["project"]["dependencies"]:
        parsed_dep = packaging.requirements.Requirement(dep)
        if parsed_dep.name == package_name:
            supported_versions: List[str] = []

            for release_version in versions:
                if packaging.version.parse(release_version) in parsed_dep.specifier:
                    supported_versions.append(release_version)

            return supported_versions

    exit_error(
        f"{package_name} is not defined in dependency section.\n"
        + f"{'\n'.join(pyproject_toml['project']['dependencies'])}"
    )

    return []


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


@typeguard.typechecked
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


@typeguard.typechecked
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


@typeguard.typechecked
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
                    "CI_CONTAINER_NAME": "ubuntu24.04",
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
# If a package is not define in the list, but defined in the pyproject.toml,
# pip decides which version is used.
PACKAGES_TO_TEST: Dict[str, Callable] = {
    "typeguard": get_all_major_pypi_versions,
    "jsonschema": get_all_major_pypi_versions,
    "picmistandard": get_all_pypi_versions,
    "pydantic": get_all_major_pypi_versions,
    "referencing": get_all_major_pypi_versions,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit_error("Pass path to Project.toml as first argument.")

    # read pyproject.toml
    with open(sys.argv[1], "rb") as f:
        pyproject_toml = tomllib.load(f)

    # the key is the name of the package and the value are all versions to be tested
    test_pkg_versions: Dict[str, List[str]] = {}

    # pull release versions from pypy.org
    # depending on the version_strategy maybe all versions are crawled or less, like all major
    # releases
    for pkg, version_strategy in PACKAGES_TO_TEST.items():
        test_pkg_versions[pkg] = get_all_pypy_releases(pkg, version_strategy)

    # remove all release version, which are not supported by the version range configured in the
    # pyproject.toml
    for pkg in test_pkg_versions:
        test_pkg_versions[pkg] = get_supported_versions(pkg, test_pkg_versions[pkg], pyproject_toml)

    print_job_yaml(test_pkg_versions)
