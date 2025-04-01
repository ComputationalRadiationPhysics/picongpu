"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Brian Edward Marre, Masoud Afshari, Julian Lenz
License: GPLv3+
"""

from ..rendering import SelfRegisteringRenderedObject


import typeguard


@typeguard.typechecked
class Plugin(SelfRegisteringRenderedObject):
    """general interface for all plugins"""

    pass
