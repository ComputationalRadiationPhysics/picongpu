"""
This file is part of the PIConGPU.

Copyright 2017-2020 PIConGPU contributors
Authors: Sebastian Starke
License: GPLv3+
"""

import os


def get_all_dirs_with_prefix(path, prefix):
    """
    Returns all directories starting with a given prefix within the given
    path
    """
    dirs = [d for d in os.listdir(
        path) if os.path.isdir(os.path.join(path, d))]

    prefix_filtered = [d for d in dirs if d.startswith(prefix)]
    return sorted(prefix_filtered)


def get_all_scans(path):
    """
    """
    return get_all_dirs_with_prefix(path, "scan_")


def get_all_sims(path):
    """
    """
    return get_all_dirs_with_prefix(path, "sim_")
