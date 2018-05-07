"""
This file is part of PIConGPU.

Copyright 2017-2018 PIConGPU contributors
Authors: Sebastian Starke, Jeffrey Kelling
License: GPLv3+
"""
import collections
import pint
ureg = pint.UnitRegistry()


class Parameter(object):
    """
    Parameter that can have a corresponding representation as a jupyter widget
    within interactive jupyter notebooks.
    The widget allows the user to adjust values as he/she likes but this class
    controls the true value that will be passed to PIConGPU. This is necessary
    since possibly internal values (the pic-scale) and the scale presented in
    the UI might differ
    (e.g.
    user adjusts simulation time which is internally handled as discrete steps
        (--> continuous to discrete conversion)
    or
    user adjusts the exponent of magnitude of particles but internally the
    precise number is needed (--> log scale conversion)
    or
    user adjusts time in nanoseconds but internally the precise number is
    needed (--> linear conversion)
    )
    Also the kind of widget representation should depend on the attributes of
    this class. Either as MultiRangeSlider or as MultiSelection or as Checkbox
    depending on the 'values' (discrete) or 'range' (continuous)

    Stepping will be handled by the widget representation, not here.
    """

    def __init__(self, name, ptype, unit, default,
                 values=None, range=None,
                 label=None, pic_to_SI=lambda x: x,
                 pic_from_SI=lambda x: x):
        """
        Parameters
        ----------
        name: string
            name of the parameter object
        ptype: string
            The type of parameter object, should be one of 'compile' or 'run'
            to mark parameter as PIConGPU run/compile-time parameter
        unit: string
            The measurement unit of this parameter as shown on the UI side.
            Implicitely we assume the pic unit is SI. If this is not possible,
            the converter needs to translate those scales.
        default: float, int or string
            The default value for this parameter (on the UI scale) used when
            no 'values' or 'range' parameter is passed.
        values: list
            list of discrete options for the parameter as shown on UI side.
            Only one of the attributes range/value can be given.
            Due to rounding issues, the values are only approximate.
        range: tuple
            start and stop value for the selectable range on UI side.
            Only one of the attributes range/value can be given.
            Due to rounding issues, the range is only approximate.
        label: string [optional]
            Overwrite the name for UI
        pic_to_SI: callable, e.g. lambda function
            Specifies how to transform a value on the pic scale to an
            appropriate SI value. Usually given via a Converter object.
        pic_from_SI: callable, e.g. lambda function
            Specifies how to transform an SI value to the internal pic scale.
        """

        self.name = name
        self.type = ptype
        self.unit = ureg.parse_units(unit)
        self.base_unit = ureg.get_base_units(self.unit)[1]
        self.default = default
        self.pic_to_SI = pic_to_SI
        self.pic_from_SI = pic_from_SI

        # for slider widget creation
        self.label = label or name

        self.range = None
        self.values = None

        if values is not None and range is not None:
            raise ValueError("Can only set either 'values' or 'range'!")
        elif values is not None:
            if not isinstance(values, collections.Iterable):
                values = [values]
            if not values:
                # check empty values list
                self.values = [self.default]
                print("WARNING: Values attribute can not be an empty "
                      "iterable! Setting values to", self.values)
            # double conversion to avoid rounding issues
            self.values = self.convert_from_PIC(
                self.convert_to_PIC(self.values))
        elif range is not None:
            if len(range) != 2:
                raise ValueError("Range needs to be a tuple of length 2!")
            else:
                self.range = tuple(self.convert_from_PIC(
                    self.convert_to_PIC(range)))
        else:
            # raise ValueError("Need either 'values' or 'range' parameter!")
            self.values = self.convert_from_PIC(
                self.convert_to_PIC([self.default]))
            print("WARNING: Neither 'values' nor 'range' was given, setting"
                  " 'values' to ", self.values)

    def _check_input(self, vals):
        """
        For values that are assumed to be on the UI scale (i.e. with unit =
        self.unit), checks whether they are in the allowed range or the
        allowed discrete values.
        Raises a ValueError if a value value outside the range is detected.

        Parameters
        ----------
        vals: float or list of floats
            values on the parameters UI scale which will be checked.
        """
        if self.values is not None:
            # check for valid values
            res = all([v in self.values for v in vals])
            if not res:
                raise ValueError(
                    "Invalid values found! Values should be elements of "
                    "self.values!")
        else:
            # check for valid range
            res = all([self.range[0] <= v <= self.range[1] for v in vals])
            if not res:
                raise ValueError("Invalid values found! Values should be "
                                 "contained in self.range!")

    def convert_to_PIC(self, vals, check_vals=False):
        """
        Takes values in UI units, converts them to SI and after that
        to quantities used within PIConGPU.

        Parameters
        ----------
        vals: float or list of floats
            values that will be converted to the PIC scale
        check_vals: bool
            Flag to decide whether given values are within the allowed
            range for this parameters UI scale.

        Returns
        -------
        A list of converted values.
        """
        if check_vals:
            self._check_input(vals)

        return [self.pic_from_SI(
            (v * self.unit).to_base_units().magnitude) for v in vals]

    def convert_from_PIC(self, vals, check_vals=False):
        """
        Takes PIC values and returns values on UI scale after converting to SI
        values as intermediate step.

        Parameters
        ----------
        vals: float or list of floats
            values that will be converted to the UI scale.

        check_vals: bool
            Flag to decide whether converted values are within the allowed
            range for this parameters UI scale.

        Returns
        -------
        A list of converted values.
        """
        # v is given in PIC quantity, so we convert to UI unit
        ui_results = [
            ureg.convert(self.pic_to_SI(v), self.base_unit, self.unit)
            for v in vals]

        if check_vals:
            self._check_input(ui_results)
        return ui_results

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
