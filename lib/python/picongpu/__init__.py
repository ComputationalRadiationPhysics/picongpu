# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2026 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from . import core as core
from ._rc_params import DirtyResetError as DirtyResetError
from ._rc_params import rc_params as rc_params
from ._version import __version__ as __version__
