"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz, Masoud Afshari
License: GPLv3+
"""

import typeguard


@typeguard.typechecked
class TimeStepSpec:
    """
    Specification for time steps to perform output at in PIConGPU.

    Contains a list of slices which define at which time steps to perform output.
    Slices are expected to be normalized (i.e., non-negative, step > 0).
    """

    specs = None

    def __init__(self, specs: list[slice]):
        self.specs = specs

    def get_rendering_context(self) -> dict:
        """
        Get the rendering context as expected by the PIConGPU backend.

        :return: dict with specs as list of dicts, each containing start, stop, step
        """
        return {
            "specs": [
                {
                    "start": spec.start,
                    "stop": spec.stop if spec.stop == spec.start + 1 and spec.step == 1 else spec.stop - 1,
                    "step": spec.step,
                }
                for spec in self.specs
                if spec.start < spec.stop
            ]
        }
