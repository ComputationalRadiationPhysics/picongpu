"""
This file is part of the PIConGPU.

Copyright 2022-2023 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+
"""

import argparse
import os
import sys

# add the testsuite package to the path
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)) + "/../../")
from testsuite import _manager as manager  # noqa

parser = argparse.ArgumentParser(
    description="Starts the test suite."
    " The complete Data.py cannot currently"
    " be replaced with the parameter transfer."
    " So far, only variable parameters can be"
    " passed. Please refer to the Data.py"
    " documentation for more information."
)

parser.add_argument(
    "-r",
    help="Path of the folder where the results" " of the test-suite should be saved",
    dest="result",
    type=str,
    const=None,
    nargs="?",
    default=None,
)

parser.add_argument(
    "-s",
    help="Path of the folder in which the results" " of the simulation were saved",
    dest="data",
    type=str,
    const=None,
    nargs="?",
    default=None,
)

parser.add_argument(
    "-p",
    help="Path of the folder in which the parameter" " files .params of the simulation were saved.",
    dest="param",
    type=str,
    const=None,
    nargs="?",
    default=None,
)

parser.add_argument(
    "-o",
    help="Path of the folder in which the files for" " openPMD evaluation can be found.",
    dest="openPmd",
    type=str,
    const=None,
    nargs="?",
    default=None,
)

parser.add_argument(
    "-j",
    help="Path of the folder to the json files" " if used.",
    dest="json",
    type=str,
    const=None,
    nargs="?",
    default=None,
)

args = parser.parse_args()

# start testsuite with all parameter
manager.run_testsuite(
    dataDirection=args.data,
    paramDirection=args.param,
    jsonDirection=args.json,
    resultDirection=args.result,
)
