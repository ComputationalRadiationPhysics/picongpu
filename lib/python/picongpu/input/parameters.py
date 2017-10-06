"""
This file is part of PIConGPU.

Copyright 2017 PIConGPU contributors
Authors: Sebastian Starke, Jeffrey Kelling
License: GPLv3+
"""

from numpy import log10


class Parameter(object):
    def __init__(self, name, ptype, unit, default, value=None):
        self.name = name
        self.type = ptype  # decides if compile time or runtime parameter
        self.unit = unit
        self.default = default

        self.value = value if value is not None else default

    def __str__(self):
        s = ""
        s += "name=" + self.name
        s += "\ttype=" + self.type
        s += "\tunit=" + self.unit
        s += "\tdefault=" + str(self.default)
        s += "\tvalue=" + str(self.value)
        return s

    def as_dict(self):
        return self.__dict__

    def macro_name(self):
        return "PARAM_" + self.name.upper()

    def dict_name(self):
        return self.name.upper()

    def new_with_default(self):
        return Parameter(self.name, self.type, self.unit, self.default)


class UiParameter(Parameter):

    def __init__(self, name, ptype, unit, default, slider_min, slider_max,
                 slider_step, value=None):
        Parameter.__init__(self, name, ptype, unit, default, value)

        # for slider widget creation
        self.min = slider_min
        self.max = slider_max
        self.step = slider_step

    def set_value(self, x):
        """Used as callback function for ipython widget sliders."""
        self.value = x
        return self.value

    def get_value(self):
        return self.value


class LinearScaledParameter(UiParameter):

    def __init__(self, name, ptype, unit, default, slider_min, slider_max,
                 slider_step, scale_factor, value=None):

        UiParameter.__init__(self, name, ptype, unit, default,
                             slider_min, slider_max, slider_step, value)

        self.scale_factor = scale_factor

    def set_value(self, x):
        """Set value to a linear scaled version of the slider value """
        self.value = x * self.scale_factor
        return self.value

    def get_value(self):
        """Return the slider value from self.value by reverting the scaling"""
        return self.value / self.scale_factor


class LogScaledParameter(Parameter):

    def __init__(self, name, ptype, unit, default, slider_min, slider_max,
                 slider_step, base, value=None):

        UiParameter.__init__(self, name, ptype, unit, default,
                             slider_min, slider_max, slider_step, value)

        self.base = float(base)

    def set_value(self, x):
        """Set value to a power version of the slider value """
        self.value = self.base ** x
        return self.value

    def get_value(self):
        """Return the slider value from self.value by log transform"""
        # using numpy gives better precision
        # return log(self.value, self.base)
        # uses the fact that log_b(v) = log_10(v) / log_10(b)
        return log10(self.value) / log10(self.base)
