"""Copyright 2023 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Verification of the results.
"""

from typing import List, Dict, Tuple, Union
from typeguard import typechecked
from packaging import version as pk_version

from alpaka_job_coverage.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from alpaka_job_coverage.versions import is_supported_version
from util import print_warn


def verify_parameters(parameters: Dict[str, Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]]]):
    """Prints a warning for each parameter value which is not supported by the
    alpaka-job-coverage library.

    Args:
        parameters (Dict[str, Union[List[Tuple[str, str]], List[List[Tuple[str, str]]]]]):
            All combination parameters.
    """
    for param_name, param_value in parameters.items():
        if param_name == BACKENDS:
            for backend in param_value:
                for name, version in backend:
                    if not is_supported_version(name=name, version=version):
                        print_warn(f"{name}-{version} is not officially supported by " "the alpaka-job-library.")
        elif param_name not in [BUILD_TYPE, JOB_EXECUTION_TYPE, MDSPAN]:
            for name, version in param_value:
                # if we compare a minor.major.patch version with a minor.major
                # version, the check is only true, if all three numbers matches
                # e.g. 2.4 == 2.4.0 -> True
                #      2.4 == 2.4.1 -> False
                if name == CMAKE:
                    parsed_cmake_version = pk_version.parse(version)
                    mod_version = f"{parsed_cmake_version.major}.{parsed_cmake_version.minor}"
                else:
                    mod_version = version
                if not is_supported_version(name=name, version=mod_version):
                    print_warn(f"{name}-{mod_version} is not officially supported by " "the alpaka-job-library.")


class Combination:
    def __init__(self, parameters: Dict[str, Tuple[str, str]]):
        """A combination describes a (sub-)set of parameters and has a state
        found (default false). If all parameters defined in the combination are
        contained in a row, the found state is changed to true.

        Args:
            parameters (Dict[str, Tuple[str, str]]): Set of parameter which
            should be found
        """
        self.parameters = parameters
        self.found = False

    def match_row(self, row: Dict[str, Tuple[str, str]]) -> bool:
        """Check if all parameters are contained in the row. If all parameters
        are found, change internal found state to True.

        Args:
            row (Dict[str, Tuple[str, str]]): The row

        Returns:
            bool: Return True, if all parameters was found.
        """
        for param_name, name_version in self.parameters.items():
            name, version = name_version
            # use * as wildcard and test only the name
            if version == "*":
                if row[param_name][0] != name:
                    return False
            else:
                if row[param_name] != name_version:
                    return False

        self.found = True
        return True

    def __str__(self) -> str:
        s = ""
        if self.found:
            s += "\033[32mfound: "
        else:
            s += "\033[31mnot found: "

        s += str(self.parameters)

        s += "\033[m"
        return s


@typechecked
def verify(combinations: List[Dict[str, Tuple[str, str]]]) -> bool:
    """Check if job matrix fullfill certain requirements.
    Args:
        combinations (List[Dict[str, Tuple[str, str]]]): The job matrix.

    Returns:
        bool: True if all checks passes, otherwise False.
    """

    #############################################################
    # check if the combinations are created
    #############################################################
    combinations_to_search = [
        # use * as wildcard for the version and test if the compiler
        # combinations exists
        Combination({HOST_COMPILER: (GCC, "*"), DEVICE_COMPILER: (NVCC, "*")}),
        Combination({HOST_COMPILER: (CLANG, "*"), DEVICE_COMPILER: (NVCC, "*")}),
        Combination({HOST_COMPILER: (HIPCC, "*"), DEVICE_COMPILER: (HIPCC, "*")}),
        Combination({HOST_COMPILER: (CLANG_CUDA, "*"), DEVICE_COMPILER: (CLANG_CUDA, "*")}),
        Combination({DEVICE_COMPILER: (NVCC, "12.0"), CXX_STANDARD: (CXX_STANDARD, "20")}),
        Combination({DEVICE_COMPILER: (NVCC, "12.1"), CXX_STANDARD: (CXX_STANDARD, "20")}),
        Combination({HOST_COMPILER: (ICPX, "*"), DEVICE_COMPILER: (ICPX, "*")}),
    ]

    for cs in combinations_to_search:
        for row in combinations:
            if cs.match_row(row):
                break

    missing_combination = False

    for cs in combinations_to_search:
        if not cs.found:
            print(cs)
            missing_combination = True

    if missing_combination:
        print("\033[31mverification failed\033[m")
        return False

    print("\033[32mverification passed\033[m")
    return True
