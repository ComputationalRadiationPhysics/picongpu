"""
This file is part of PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sebastian Starke, Jeffrey Kelling
License: GPLv3+
"""

import numpy as np
import inspect


class Parameter(object):
    """
    Base class for exposable parameters.
    """

    def __init__(self, name, ptype, unit, default, value=None, dtype=None):
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
        dtype: type (optional)
            The data type of the parameter value. If it is not provided, the
            type of the 'default' argument is used.
        """
        self.name = name
        self.type = ptype
        self.unit = unit
        self.default = default

        self.dtype = dtype if dtype is not None else type(default)
        # use the dtype to cast the value to the correct type
        self.value = self.dtype(
            value) if value is not None else self.dtype(default)

    def as_dict(self):
        """
        Convert Parameter object into a plain dictionary

        Returns
        --------
        A dictionary with the member variables as (key, value) pairs.
        """
        d = dict(
            cls=type(self).__name__,
            name=self.name,
            type=self.type,
            unit=self.unit,
            default=self.default,
            value=self.value,
            dtype=self.dtype.__name__)

        return d

    def __str__(self):
        """
        Enable use of print() function with Parameter objects.

        Returns
        -------
        A string representation of the Parameter objects member values.
        """
        return self.as_dict().__str__()

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
                 slider_step, label=None, formatter=None,
                 value=None, dtype=None):
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
        label: string [optional]
            Overwrite the name for UI
        formatter: function or lambda of form f(x) -> string [optional]
            representation of the current value for UI

        """
        Parameter.__init__(self, name, ptype, unit, default, value, dtype)

        # for slider widget creation
        self.min = slider_min
        self.max = slider_max
        self.step = slider_step

        if label is None:
            self.label = name
        else:
            self.label = label
        if formatter is None:
            self.formatter = lambda x: str(x)
        else:
            self.formatter = formatter

        # explicitely call the set_value function to handle
        # scaling and transformation in derived classes
        # already on initialization. if this was missing, then
        # parameter values would possibly be on the wrong scale
        self.set_value(self.value)

    def convert_value_to_internal_scale(self, x):
        """
        Computes for a given value x its corresponding
        representation on the 'internal scale' of the parameter
        after taking into account the possible transformation operation.
        For UiParameters it just returns the identity, but derived
        classes might perform e.g. scaling or exponential transformation
        to convert values to the internal scale of the parameters
        for usage within picongpu.

        Parameters
        ----------
        x: float
            The value whose corresponding representation
            needs to be computed.

        Returns
        -------
        The value cast to the parameters dtype
        """

        return self.dtype(x)

    def set_value(self, x):
        """
        Updates the objects 'value' attribute to (a possibly transformed)
        internal representation

        Parameters
        ----------
        x: float
            The current value of the slider widget

        Returns
        -------
        A float containing the updated 'value' attribute. Is printed
        below the slider-widget for user-feedback.
        """

        self.value = self.convert_value_to_internal_scale(x)
        return self.formatter(self.value)

    def on_value_change(self, change):
        """
        Callback function used with observe() function of ipython widgets.
        Just a wrapper of the set_value function.

        Parameters
        ----------
        change: dict
                  As passed by the Ipython widgets framework.

        Returns
        -------
        Whatever the set_value function returns.
        """

        if change["type"] == "change":
            return self.set_value(x=change["new"])

    def convert_value_from_internal_scale(self, x):
        """
        Used for value conversion from the parameters internal scale
        to the ui scale
        It computes the inverse of the transformation
        that is used in convert_value_to_internal_scale().

        Since those scales are the same for UiParameter
        objects, this computes the identity function.

        In derived classes however, there might exist a
        transformation function to convert between ui scale
        values and internal parameter scale values
        (e.g. slider shows values between
        3 and 6, but the internal parameter scale is
        between 3.e-9 and 6.e-9).

        Parameters
        ----------
        x: float
            a value on the same scale as the parameters self.value

        Returns
        -------
        the outcome of the identity function since the ui
        scale and the internal scale are identical.
        """

        return x

    def get_value_on_ui_scale(self):
        """
        Function that reverts a possible transformation of ui
        value to internal value.
        Since for UiParameter there is no conversion, it returns
        the self.value.
        In derived classes however, there will be a reverse
        transformation of the self.value to the scale of the ui range
        (which e.g. the slider will use for displaying purposes)

        Returns
        -------
        The self.value after conversion to the ui scale
        """
        return self.convert_value_from_internal_scale(self.value)

    def as_dict(self):
        """
        Convert Parameter object into a plain dictionary

        Returns
        --------
        A dictionary with the member variables as (key, value) pairs.
        """

        members = super(UiParameter, self).as_dict()
        members["label"] = self.label
        members["formatter"] = str(inspect.getsourcelines(self.formatter)[
                                   0]).strip("['\\n']").split(" = ")[1]

        return members


class LinearScaledParameter(UiParameter):
    """
    Parameter that can be displayed in jupyter notebooks but whose internal
    'value' is a linear transformation of the slider values.
    """

    def __init__(self, name, ptype, unit, default, slider_min, slider_max,
                 slider_step, scale_factor=1.0, label=None, formatter=None,
                 value=None, dtype=None):
        """
        Parameters
        ----------
        In addition to base class parameters:

        scale_factor: float
            The factor for the linear transformation of slider ui value to
            internal value.
        """
        self.scale_factor = scale_factor
        # since base class UiParameter now calls the set_value() to carry out
        # linear scaling of the value, we need to define the scale_factor
        # before entering the constructor since it is used in set_value()

        UiParameter.__init__(self, name, ptype, unit, default,
                             slider_min, slider_max, slider_step,
                             label, formatter, value, dtype)

    def convert_value_to_internal_scale(self, x):
        """
        Computes for a given value its corresponding
        representation on the 'internal scale' of the parameter
        after taking into account the possible linear
        scaling operation.

        Parameters
        ----------
        x: float
            The value whose corresponding representation
            needs to be computed.

        Returns
        -------
        The value multiplied by the parameters scale_factor
        and cast to the parameters dtype
        """

        return self.dtype(x * self.scale_factor)

    def convert_value_from_internal_scale(self, x):
        """
        Computes the inverse of the scaling transformation
        that is used in convert_value_to_internal_scale().
        This is intended to switch from the internal scale
        of the parameters to the scale of the ui.

        Parameters
        ----------
        x: float

        Returns
        -------
        A float containing the value (on the ui scale) that the slider needs
        to display for proper representation of the internal 'value' attribute.
        This inverts the linear scaling procedure.
        Overrides the base class method.
        """

        return float(x) / self.scale_factor

    def as_dict(self):
        """
        Convert Parameter object into a plain dictionary

        Returns
        --------
        A dictionary with the member variables as (key, value) pairs.
        """

        members = super(LinearScaledParameter, self).as_dict()
        members["scale_factor"] = self.scale_factor

        return members


class LogScaledParameter(UiParameter):
    """
    Parameter that can be displayed in jupyter notebooks but whose internal
    'value' is computed from a power transformation of the slider values.
    """

    def __init__(self, name, ptype, unit, default, slider_min, slider_max,
                 slider_step, base, label=None, formatter=None,
                 value=None, dtype=None):
        """
        Parameters
        ----------
        In addition to base class parameters:

        base: float
            The base for the power transformation from slider ui value to
            the internal 'value'.
        """
        self.base = float(base)

        UiParameter.__init__(self, name, ptype, unit, default,
                             slider_min, slider_max, slider_step,
                             label, formatter, value, dtype)

    def convert_value_to_internal_scale(self, x):
        """
        Computes for a given value its corresponding
        representation on the 'internal scale' of the parameter
        after taking into account the exponential transform operation.

        Parameters
        ----------
        x: float
            The value whose corresponding representation
            needs to be computed.

        Returns
        -------
        The value taken as power of the parameters base
        and cast to the parameters dtype
        """

        return self.dtype(self.base ** x)

    def convert_value_from_internal_scale(self, x):
        """
        Computes the inverse of the power transformation
        that is used in convert_value_to_internal_scale().
        This is intended to switch from the internal scale
        of the parameters to the scale of the ui.

        Parameters
        ----------
        x: float

        Returns
        -------
        A float containing the value (on the ui scale) that the slider
        needs to display for proper representation of the
        internal 'value' attribute.
        This inverts the power transformation by using log.
        Overrides the base class method.
        """

        # want to return log_b(x)
        # which is identical to log_10(x) / log_10(b)
        # by logarithmic law.
        return np.log10(float(x)) / np.log10(self.base)

    def as_dict(self):
        """
        Convert Parameter object into a plain dictionary

        Returns
        --------
        A dictionary with the member variables as (key, value) pairs.
        """

        members = super(LogScaledParameter, self).as_dict()
        members["base"] = self.base

        return members
