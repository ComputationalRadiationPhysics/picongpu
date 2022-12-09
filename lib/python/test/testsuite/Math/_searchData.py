"""
This file is part of the PIConGPU.

Copyright 2022 PIConGPU contributors
Authors: Mika Soren Voss
License: GPLv3+

Routines in this module:

searchParameter(parameter:str, directiontype:str = None)
"""

__all__ = ["searchParameter"]

from .. import Reader
import warnings


def searchParameter(parameter: str, directiontype: str = None, **kwargs):
    """
    searches in the data (passed in Data.py as a directory).
    To avoid errors, usage without directiontype is not recommended

    Input:
    -------

    parameter :     str
                    Designation of the parameters. Must match the
                    designations in the data files.

    directiontype : str, optional
                    "param", "json", "dat" or "openpmd".
                    If None, the parameter is searched for in all files
                    specified in Data.py.

    Raise:
    -------
    ValueError:
        if directiontype not in ["param", "json", "dat", "openpmd", None]
        if the parameter could not be found

    Return:
    -------
    out : float or array
    """

    if (directiontype not in ["param", "json", "dat", "openpmd"] and
            directiontype is not None):
        raise ValueError("directiontype must be either None, param,"
                         " dat, json, or openpmd")

    if directiontype == "param":
        result = Reader.paramReader.getValue(parameter)
    elif directiontype == "json":
        result = Reader.jsonReader.getValue(parameter)
    elif directiontype == "dat":
        result = Reader.dataReader.getValue(parameter, **kwargs)

    if directiontype is None:
        warnings.warn("The test suite now searches for the parameters"
                      " independently. To prevent this please specify"
                      " directiontype.")
        try:
            result = Reader.paramReader.getValue(parameter)
        except Exception:
            result = None

        if result is None:
            try:
                result = Reader.jsonReader.getValue(parameter)
            except Exception:
                result = None

        if result is None:
            try:
                result = Reader.dataReader.getValue(parameter)
            except Exception:
                result = None

    if result is None:
        raise ValueError("The Parameter {} could not"
                         " be found".format(parameter))

    return result
