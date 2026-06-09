# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from functools import lru_cache
from picongpu import core


@lru_cache
def _parse_version():
    with (core.path("include") / "picongpu" / "version.hpp").open("r") as version_header:
        header_code = version_header.read()
    version_dict = {
        key: s.split(" ")[1].strip()
        for key, s in zip(
            ("major", "minor", "patch", "label"),
            filter(lambda s: s.startswith("PICONGPU_VERSION_"), header_code.split("\n#define ")),
        )
    }
    return f"{version_dict['major']}.{version_dict['minor']}.{version_dict['patch']}" + (
        f"-{version_dict['label']}" if "label" in version_dict else ""
    )


__version__ = _parse_version()
