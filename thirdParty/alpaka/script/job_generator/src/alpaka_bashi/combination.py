"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Modify and add additional parameters to combinations.
"""

from copy import deepcopy
from typeguard import typechecked
import bashi
from alpaka_bashi.globals import JOB_EXECUTION_TYPE, CI_PIPELINE_NAME

from alpaka_bashi.combination_modifier.execution_type import (
    execution_type_device_compiler_gcc_and_clang,
    execution_type_hipcc,
    execution_type_icpx,
    execution_type_cuda_backend,
)
from alpaka_bashi.combination_modifier.ci_pipeline_type import add_ci_pipeline_type


@typechecked
def add_combinations_parameters(combination_list: bashi.CombinationList) -> bashi.CombinationList:
    """Add parameters and parameter-values to the combination depending on the existing
    parameter-values.

    Args:
        combination_list (bashi.CombinationList): combination-list

    Raises:
        RuntimeError: If one or more combination misses a parameter after applying the modification
        functions.
    """
    combination_list_copy = deepcopy(combination_list)

    combination_list_copy = execution_type_device_compiler_gcc_and_clang(combination_list_copy)
    combination_list_copy = execution_type_hipcc(combination_list_copy)
    combination_list_copy = execution_type_icpx(combination_list_copy)
    combination_list_copy = execution_type_cuda_backend(combination_list_copy)
    combination_list_copy = add_ci_pipeline_type(combination_list_copy)

    # count the number
    annotated_combinations = {JOB_EXECUTION_TYPE: 0, CI_PIPELINE_NAME: 0}
    num_combs = len(combination_list_copy)

    for comb in combination_list_copy:
        for parameter in annotated_combinations:
            if parameter in comb:
                annotated_combinations[parameter] += 1
            else:
                print(
                    "Parameter {parameter} missing in combinations.\n"
                    f"{bashi.get_str_row_nice(comb)}"
                )

    # check if all combinations have the required parameters
    for parameter, num_annotations in annotated_combinations.items():
        if num_annotations < num_combs:
            raise RuntimeError(
                f"{num_combs - num_annotations} of {num_combs} combinations have no parameter "
                f"{parameter}."
            )

    return combination_list_copy
