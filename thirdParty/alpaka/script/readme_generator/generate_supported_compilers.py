#!/usr/bin/env python3

import os
import sys
import json
from typing import List, Dict
from dataclasses import dataclass


def print_red(msg: str):
    """print message with color red"""
    print("\033[0;31m" + msg + "\033[0m")


@dataclass
class ConfigEntry:
    """Stores the key for the json config and how the key is printed in
    markdown.
    """

    name: str
    representation: str


# pylint: disable=missing-docstring
def get_expected_config_entries() -> List[ConfigEntry]:
    return [
        ConfigEntry("serial", "Serial"),
        ConfigEntry("OMPblock", "OpenMP 2.0+ blocks"),
        ConfigEntry("OMPthread", "OpenMP 2.0+ threads"),
        ConfigEntry("thread", "std::thread"),
        ConfigEntry("tbb", "TBB"),
        ConfigEntry("CUDAnvcc", "CUDA (nvcc)"),
        ConfigEntry("CUDAclang", "CUDA (clang)"),
        ConfigEntry("hip", "HIP (clang)"),
        ConfigEntry("sycl", "SYCL"),
    ]


# pylint: disable=missing-docstring
def get_expected_config_names() -> List[str]:
    return [n.name for n in get_expected_config_entries()]


def get_known_states() -> Dict[str, str]:
    """Returns a dict of known backend states. The key is the value in the
    config json and the value how it will be printed in markdown.
    """
    return {"yes": ":white_check_mark:", "no": ":x:", "none": "-"}


# pylint: disable=missing-docstring
def get_known_state_names() -> List[str]:
    return list(get_known_states().keys())


def config_validator(conf: Dict[str, Dict[str, str]]) -> bool:
    """Validate the json configuration and prints errors."""
    for compiler_name, compiler_conf in conf.items():
        for expected_entry in get_expected_config_names():
            if expected_entry not in compiler_conf:
                print_red(f"[ERROR]: {compiler_name} misses entry {expected_entry}")
                return False
            if "state" not in compiler_conf[expected_entry]:
                print_red(
                    f"[ERROR]: {compiler_name}/{expected_entry} misses state entry"
                )
                return False
            if compiler_conf[expected_entry]["state"] not in get_known_state_names():
                print_red(
                    f"[ERROR]: {compiler_name}/{expected_entry}/state "
                    f"unknown state: {compiler_conf[expected_entry]['state']}"
                )
                return False

    return True


def render_table(conf):
    """Renders the configuration to a markdown table"""
    # [column][row]
    table: List[List[str]] = []

    # add backend names
    backends: List[str] = ["Accelerator Back-end"]
    for config_entry in get_expected_config_entries():
        backends.append(config_entry.representation)
    table.append(backends)

    # reads the state of each backend for each compiler and generates the cell
    # the cell contains at least a symbol for the state and can also contains
    # a comment
    for compiler_name, compiler_conf in conf.items():
        column: List[str] = [compiler_name]
        for backend in compiler_conf.values():
            value = get_known_states()[backend["state"]]
            if "comment" in backend:
                value += f" {backend['comment']}"
            column.append(value)
        table.append(column)

    # each cell in a column should have the same width
    # therefore determine the broadest cell in a column
    column_sizes: List[int] = []
    for col in table:
        size = 0
        for row in col:
            size = max(size, len(row))
        column_sizes.append(size)

    # print the table header
    print("|", end="")
    for c_num in range(len(table)):
        print(f" {table[c_num][0]:<{column_sizes[c_num]}} |", end="")
    print()

    # print the lines under the table header
    print("|", end="")
    for c_num in range(len(table)):
        print((column_sizes[c_num] + 2) * "-" + "|", end="")
    print()

    # prints each backend state cell for each compiler
    for r_num in range(1, len(table[0])):
        print("|", end="")
        for c_num in range(len(table)):
            print(f" {table[c_num][r_num]:<{column_sizes[c_num]}} |", end="")
        print()


if __name__ == "__main__":
    script_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_path, "supported_compilers.json")

    if not os.path.exists(config_path):
        print_red(f"[ERROR]: {config_path} does not exist")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not config_validator(config):
        sys.exit(1)

    render_table(config)
