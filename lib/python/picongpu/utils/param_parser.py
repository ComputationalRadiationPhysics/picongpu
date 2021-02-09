#!/usr/bin/env python3
"""
This file is part of the PIConGPU.

Copyright 2017-2021 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

import os
import json
import getopt
import sys


def read_range_file(file, values_only=True):
    """
    Read a json file for a PIC scan or simulation. Such files are generated
    by the python jupyter UI and have the following structure:
    PARAMETER_NAME: {
        "type": run (or compile)
        "values": [v1, v2]
    }

    assumed to
    Parameters
    ----------
    file: str
        path to a parameter json file to read (either from a scan
        or a single simulation)

    values_only: bool
        Flag to discard everything but the 'values' attribute for
        each parameter contained in the json file

    Returns
    -------
    A dictionary with keys = parameter names.
    Values are either a dictionary with 'values', 'type' and 'macro_name'
    keys for each parameter if 'values_only' is False.
    Or simply a list of values for this parameter when 'values_only' is
    True.
    """
    if not os.path.isfile(file):
        raise IOError("File {} does not exist".format(file))

    with open(file, 'r') as json_file:
        range_dict = json.load(json_file)
    if values_only:
        filtered_range_dict = dict()
        for name, attrs in range_dict.items():
            filtered_range_dict[name] = attrs['values']

        return filtered_range_dict
    else:
        return range_dict


def to_macro_name(name):
    """
    Convert a parameter name to the corresponding name in the picongpu
    compiler-define.
    """
    if not name.startswith("_"):
        return "PARAM_" + name.upper()
    else:
        return "PARAM" + name.upper()


def parse(file, ptype):
    """
    parses json files of scans or simulations for their parameters
    type (either compiletime or runtime, coded as compile/run) and
    values.

    Returns
    -------
    an argument string which is further processed by
    pic-configure with -c option for compile time parameters
    or by tbg for runtime parameters.
    """
    range_dict = read_range_file(file, values_only=False)
    # filter for correct ptype
    filtered_dict = dict()
    for param_name, attrs in range_dict.items():
        if attrs['type'] == ptype:
            # name, value mapping
            filtered_dict[param_name] = attrs['values'][0]

    # construct the statement passed to picongpu
    if ptype == "compile":
        ostr = [to_macro_name(name) + "=" + str(value)
                for name, value in filtered_dict.items()]
        cxx_defines = ";".join(map(lambda s: "-D" + s, ostr))
        return "-DPARAM_OVERWRITES:LIST='" + cxx_defines + "'"

    elif ptype == "run":
        ostr = [str(name) + "=" + str(value)
                for name, value in filtered_dict.items()]
        if not ostr:
            return ""
        else:
            return "-o " + "'" + " ".join(ostr) + "'"


if __name__ == '__main__':

    type = ''
    file = ''

    try:
        opts, args = getopt.getopt(sys.argv[1:], "ht:i:", ["type=", "ifile="])
    except getopt.GetoptError:
        print("paramParser.py -t <parameterType -i <inputfile>")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("paramParser.py -t <parameterType -i <inputfile>")
        elif opt in ("-t", "--type"):
            type = arg
        elif opt in ("-i", "--ifile"):
            file = arg

    if(type in ["compile", "run"]):
        print(parse(file, type))
    else:
        print("-t option not understood! Either choose 'compile' or 'run'!")
        sys.exit(2)
