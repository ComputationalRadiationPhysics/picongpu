"""
This file is part of PIConGPU.
Copyright 2023 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from .....rendering import RenderedObject
import typeguard


@typeguard.typechecked
class PlasmaRamp(RenderedObject):
    """
    abstract parent class for all plasma ramps

    A plasma ramp describes ramp up of an edge of an initial density
    distribution
    """

    def __init__(self):
        raise NotImplementedError()

    def check(self) -> None:
        """
        check self, overwritten by child class

        Perform checks if own parameters are valid.
        passes silently if everything is okay
        """
        foundPreviousActive = False
        foundPreviousActiveType = ""
        for typeEntry, marker in self.returned_context["type"].items():
            if marker:
                if foundPreviousActive:
                    raise ValueError(
                        "only one type may be marked as present!,"
                        + " both "
                        + foundPreviousActiveType
                        + " and "
                        + typeEntry
                        + " are marked as"
                        + "present"
                    )
                else:
                    foundPreviousActive = True
                    foundPreviousActiveType = typeEntry

    def get_generic_profile_rendering_context(self) -> dict:
        """
        retrieve a context valid for "any profile"

        **Problem:** Plasma Ramps are polymorph, there are several distinct
        implementations, each of them has its respective schema.

        In the template we need know which exact sub type of the general
        abstract type was used to generate the correct code.

        This is difficult in JSON (particularly in a mustache-compatible way)
        since no type information is available in the schema.

        **Solution:** We store which type was actually used in a wrapper
        in addition to actual data,
            provided as usual by get_rendering_context() by the plasma ramp
            instance

        If a generic plasma ramp is requested we us the wrapper schema for this
        class which contains the **type** meta information and the **data**
        content

        E.g.:

        .. code::

            {
                "type": {
                    "uniform": true,
                    "gaussian": false,
                    ...
                },
                "data": DATA
            }

        where DATA is the serialization as returned by get_rendering_context().

        There are *two* context serialization methods for density profiles:

        - get_rendering_context()
            - provided by RenderedObject parent class, serialization ("context
              building") performed by _get_serialized()
            - _get_serialized() implemented in *every plasma ramp*
            - checks against schema of respective plasma ramp
            - returned context is a representation of
                    *exactly this plasma ramp*
            - (left empty == not implemented in parent PlasmaRamp)

        - get_generic_profile_rendering_context()
            - implemented in parent PlasmaRamp
            - returned representation is generic for *any plasma ramp*
              (i.e. contains which type is actually used)
            - passes information from get_rendering_context() through
            - returned representation is designed for easy use with templating
              engine mustache
        """
        # import ramps here to avoid circular inclusion
        from .exponential import Exponential
        from .none import None_

        template_name_by_type = {Exponential: "exponential", None_: "none"}

        if self.__class__ not in template_name_by_type:
            raise RuntimeError("unkown type: {}".format(self.__class__))

        # create type meta data dict
        # init with all false
        type_dict = dict(map(lambda type_name: (type_name, False), template_name_by_type.values()))
        # set this class entry to true
        self_class_template_name = template_name_by_type[self.__class__]
        type_dict[self_class_template_name] = True

        # get data from actual inheriting implementation
        serialized_data = self.get_rendering_context()

        # final context to be returned: data + type info
        self.returned_context = {
            "type": type_dict,
            "data": serialized_data,
        }
        self.check()

        # make sure it passes schema checks
        RenderedObject.check_context_for_type(PlasmaRamp, self.returned_context)

        return self.returned_context
