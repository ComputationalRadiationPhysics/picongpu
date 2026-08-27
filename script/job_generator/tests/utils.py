"Util functions for unit tests."

from typing import cast, List, Tuple, TypeAlias
from collections import OrderedDict
from itertools import chain
import packaging.version
import bashi
import bashi.utils

ParsableVersion: TypeAlias = str | int | float | bashi.ValueVersion
CompilerParsableParameterValue: TypeAlias = Tuple[bashi.Parameter, bashi.ValueName, ParsableVersion]
DefaultParsableParameterValue: TypeAlias = Tuple[bashi.Parameter, ParsableVersion]
ParsableParameterValue: TypeAlias = DefaultParsableParameterValue | CompilerParsableParameterValue


def parse_param_val(
    param_val: Tuple[bashi.ValueName, ParsableVersion],
) -> bashi.ParameterValue:
    """Parse a single tuple to a parameter-values.

    Args:
        param_val (Tuple[ValueName, ParsableVersion]): Tuple to parse

    Returns:
        ParameterValue: parsed ParameterValue
    """
    val_name, val_version = param_val
    if isinstance(val_version, bashi.ValueVersion):
        parsed_version = val_version
    else:
        parsed_version = packaging.version.parse(str(val_version))
    return bashi.ParameterValue(val_name, parsed_version)


def parse_param_vals(
    param_vals: List[Tuple[bashi.ValueName, ParsableVersion]],
) -> List[bashi.ParameterValue]:
    """Parse a list of tuples to a list of parameter-values.

    Args:
        param_vals (List[Tuple[ValueName, ParsableVersion]]): List to parse

    Returns:
        List[ParameterValue]: List of parameter-values
    """
    parsed_list: List[bashi.ParameterValue] = [
        parse_param_val(param_val) for param_val in param_vals
    ]

    return parsed_list


def parse_param_value_tuples(input_list: List[ParsableParameterValue]) -> bashi.ParameterValueTuple:
    """Parse a list of tuples to a parameter-value-tuple.

    Args:
        input_list (List[Tuple[ParsableParameterValue, ParsableParameterValue]]):
            e.g.:
            parse_param_value_tuples(
            [
                (HOST_COMPILER, GCC, 12), (UBUNTU, 22.04), (DEVICE_COMPILER, GCC, 12),
                (CMAKE, "3.19"), (ALPAKA_ACC_GPU_CUDA_ENABLE, "10.1"),
                (ALPAKA_ACC_ONEAPI_CPU_ENABLE, ON)
            ])

    Raises:
        TypeError: If a host or device compiler has no ValueName and ValueVersion
        TypeError: If a parameter which is not host or device compiler has no ValueVersion

    Returns:
        bashi.ParameterValueTuple: parameter-value-tuple
    """
    row: bashi.ParameterValueTuple = OrderedDict()
    for entry in input_list:
        param_name: bashi.Parameter = entry[0]
        if param_name in (bashi.HOST_COMPILER, bashi.DEVICE_COMPILER):
            if len(entry) < 3:
                raise TypeError(f"Parameter is {param_name}. Has not 3 values: {entry}")
            value_name = entry[1]
            value_version = entry[2]
        else:
            if len(entry) < 2:
                raise TypeError(f"Has not 2 values: {entry}")
            value_name = param_name
            value_version = entry[1]
        row[param_name] = parse_param_val((value_name, value_version))

    return row


def parse_bashi_row(input_list: List[ParsableParameterValue]) -> bashi.BashiRow:
    """Parse a list of tuples to a BashiRow. See parse_param_value_tuples()

    Returns:
        bashi.BashiRow: parameter-value-tuple
    """

    return bashi.BashiRow(parse_param_value_tuples(input_list))


RegularParsableParameterValue: TypeAlias = Tuple[bashi.Parameter, bashi.ValueName, ParsableVersion]


def parse_expected_val_pairs(
    input_list: List[Tuple[ParsableParameterValue, ParsableParameterValue]],
) -> List[bashi.ParameterValuePair]:
    """Parse list of expected parameter-values to the correct type.

    Args:
        input_list (List[Tuple[ParsableParameterValue, ParsableParameterValue]]):
            e.g.:
            parse_expected_val_pairs(
            [
                ((HOST_COMPILER, GCC, 12), (UBUNTU, 22.04)),
                ((HOST_COMPILER, CLANG_CUDA, 14), (CMAKE, "3.19")),
                ((UBUNTU, "20.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, "10.1")),
                ((UBUNTU, "22.04"), (ALPAKA_ACC_GPU_CUDA_ENABLE, "10.1")),
            ])

    Returns:
        List[ParameterValuePair]: Parsed parameter-type list
    """
    expected_val_pairs: List[bashi.ParameterValuePair] = []
    for pair_number, input_pair in enumerate(input_list):
        regular_entry_pair: List[RegularParsableParameterValue] = []
        for entry_number, input_entry in enumerate(input_pair):
            if input_entry[0] in (bashi.HOST_COMPILER, bashi.DEVICE_COMPILER):
                compiler_input_entry = cast(CompilerParsableParameterValue, input_entry)
                if len(compiler_input_entry) != 3:
                    raise ValueError(
                        f"input_list[{pair_number}][{entry_number}] {compiler_input_entry}\n"
                        "First value is HOST_COMPILER or DEVICE_COMPILER.\n"
                        "Therefore the tuple needs to contain three entries:"
                        "\n(<HOST_COMPILER|DEVICE_COMPILER>, <value-name>, <value-version>)"
                    )
                regular_entry_pair.append(compiler_input_entry)
            else:
                if len(input_entry) != 2:
                    raise ValueError(
                        f"input_list[{pair_number}][{entry_number}] {input_entry}\n"
                        "The tuple needs to contain two entries:"
                        "\n(<parameter>, <value-version>)"
                    )
                default_input_entry = cast(DefaultParsableParameterValue, input_entry)
                regular_entry_pair.append(
                    (default_input_entry[0], default_input_entry[0], default_input_entry[1])
                )

        expected_val_pairs.append(
            bashi.utils.create_parameter_value_pair(*chain(*regular_entry_pair))
        )

    return expected_val_pairs
