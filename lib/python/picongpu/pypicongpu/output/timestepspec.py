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

    def get_rendering_context(self, num_steps=200) -> dict:
        """
        Get the rendering context as expected by the PIConGPU backend.

        :param num_steps: Total number of simulation steps (default: 200).
        :return: dict with specs as list of dicts, each containing start, stop, step
        """
        specs = []
        for spec in self.specs:
            start = spec.start if spec.start is not None else 0
            stop = spec.stop if spec.stop is not None else num_steps
            step = spec.step if spec.step is not None else 1
            if start < stop:
                specs.append(
                    {
                        "start": start,
                        "stop": stop if stop == start + 1 and step == 1 else stop - 1,
                        "step": step,
                    }
                )
        return {"specs": specs}
