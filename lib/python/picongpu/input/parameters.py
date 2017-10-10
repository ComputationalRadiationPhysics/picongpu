"""
This file is part of PIConGPU.

Copyright 2017 PIConGPU contributors
Authors: Sebastian Starke, Jeffrey Kelling
License: GPLv3+
"""

from numpy import log10


class Parameter(object):
    """
    Base class for exposable parameters.
    """

    def __init__(self, name, ptype, unit, default, value=None):
        """
        Parameters
        ----------
        name: string
            name of the parameter object
        ptype: string
            The type of parameter object, should be one of 'compile' or 'run'
            to mark parameter as PIConGPU run/compile-time parameter
        unit: string
            The measurement unit of this parameter
        default: float
            The default value for this parameter used when no 'value' parameter
            is passed.
        value: float (optional)
            The value of the parameter object. If it is not provided, the value
            of the 'default' argument is used.
        """
        self.name = name
        self.type = ptype  # decides if compile time or runtime parameter
        self.unit = unit
        self.default = default

        self.value = value if value is not None else default

    def __str__(self):
        """
        Enable use of print() function with Parameter objects.

        Returns
        -------
        A string representation of the Parameter objects member values.
        """

        s = ""
        s += "name=" + self.name
        s += "\ttype=" + self.type
        s += "\tunit=" + self.unit
        s += "\tdefault=" + str(self.default)
        s += "\tvalue=" + str(self.value)
        return s

    def as_dict(self):
        """
        Convert Parameter object into a plain dictionary

        Returns
        --------
        A dictionary with the member variables as (key, value) pairs.
        """

        return self.__dict__

    def macro_name(self):
        """
        Returns
        -------
        A string containing the objects name as it will be used within cmake
        defines and cpp headers.
        """
        return "PARAM_" + self.name.upper()

    def dict_name(self):
        """
        Returns
        -------
        A string containing the objects name in upper case.
        """
        return self.name.upper()

    def new_with_default(self):
        """
        Returns
        -------
        A new Parameter object with the same name, type and unit but whose
        value is set to the default value.
        """
        return Parameter(self.name, self.type, self.unit, self.default)


class UiParameter(Parameter):
    """
    Parameter that can be displayed and modified through slider widgets
    in jupyter notebooks.
    """

    def __init__(self, name, ptype, unit, default, slider_min, slider_max,
                 slider_step, value=None):
        """
        Parameters
        ----------
        In addition to base class parameters:

        slider_min: float
            The minimal value of the slider widget
        slider_max: float
            The maximal value of the slider widget
        slider_step: float
            The stepsize when adjusting the slider

        """
        Parameter.__init__(self, name, ptype, unit, default, value)

        # for slider widget creation
        self.min = slider_min
        self.max = slider_max
        self.step = slider_step

    def set_value(self, x):
        """
        Callback function that updates the objects 'value' attribute
        when the corresponding ipython slider widget is modified.

        Parameters
        ----------
        x: float
            The current value of the slider widget
        Returns
        -------
        A float containing the updated 'value' attribute. Is printed
        below the slider-widget for user-feedback.
        """
        self.value = x
        return self.value

    def get_value(self):
        """
        Returns
        -------
        A float cointaining the 'value' attribute.
        """
        return self.value


class LinearScaledParameter(UiParameter):
    """
    Parameter that can be displayed in jupyter notebooks but whose internal
    'value' is a linear transformation of the slider values.
    """

    def __init__(self, name, ptype, unit, default, slider_min, slider_max,
                 slider_step, scale_factor, value=None):
        """
        Parameters
        ----------
        In addition to base class parameters:

        scale_factor: float
            The factor for the linear transformation of slider value to
            internal value.
        """
        UiParameter.__init__(self, name, ptype, unit, default,
                             slider_min, slider_max, slider_step, value)

        self.scale_factor = scale_factor

    def set_value(self, x):
        """
        Callback function that updates the internal 'value' to
        a linearly scaled version of the slider widgets value by using
        the 'scale_factor' attribute for multiplication.
        Overrides the base class method.

        Parameters
        ----------
        x: float
            The current value of the slider widget

        Returns
        -------
        A float cointaining the internal 'value' attribute after adjustment.
        """
        self.value = x * self.scale_factor
        return self.value

    def get_value(self):
        """
        Returns
        --------
        A float containing the value that the slider needs to display for
        proper representation of the internal 'value' attribute.
        This is the inverse computation of the linear scaling procedure.
        Overrides the base class method.
        """
        return self.value / self.scale_factor


class LogScaledParameter(Parameter):
    """
    Parameter that can be displayed in jupyter notebooks but whose internal
    'value' is computed from a power transformation of the slider values.
    """

    def __init__(self, name, ptype, unit, default, slider_min, slider_max,
                 slider_step, base, value=None):
        """
        Parameters
        ----------
        In addition to base class parameters:

        base: float
            The base for the power transformation from slider value to
            the internal 'value'.
        """
        UiParameter.__init__(self, name, ptype, unit, default,
                             slider_min, slider_max, slider_step, value)

        self.base = float(base)

    def set_value(self, x):
        """
        Callback function that updates the internal 'value' to
        a power version of the slider widgets value by using
        the 'base' attribute.
        Overrides the base class method.

        Parameters
        ----------
        x: float
            The current value of the slider widget

        Returns
        -------
        A float cointaining the internal 'value' attribute after adjustment.
        """

        self.value = self.base ** x
        return self.value

    def get_value(self):
        """
        Returns
        -------
        A float containing the value that the slider needs to display for
        proper representation of the internal 'value' attribute.
        This inverts the power transformation by using log.
        Overrides the base class method.
        """
        # want to return log_b(v)
        # which is identical to log_10(v) / log_10(b)
        # by logarithmic law.
        return log10(self.value) / log10(self.base)
