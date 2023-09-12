"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from typeguard import typechecked

from ....rendering import RenderedObject


@typechecked
class DensityProfile(RenderedObject):
    """
    (abstract) parent class of all density profiles

    A density profile describes the density in space.
    """

    def __init__(self):
        raise NotImplementedError()

    def check(self) -> None:
        """
        check self, overwritten by child class

        Perform checks if own parameters are valid.
        On error raise, if everything is okay pass silently.
        """
        raise NotImplementedError()

    def get_generic_profile_rendering_context(self) -> dict:
        """
        retrieve a context valid for "any profile"

        Problem: Every profile has its respective schema, and it is difficult
        in JSON (particularly in a mustache-compatible way) to get the type
        of the schema.

        Solution: The normal rendering of profiles get_rendering_context()
        provides **only their parameters**, i.e. there is **no meta
        information** on types etc.

        If a generic profile is requested one can use the schema for
        "DensityProfile" (this class), for which this method returns the
        correct content, which includes metainformation and the data on the
        schema itself.

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
            - _get_serialized() implemented in *every profile*
            - checks against schema of respective profile
            - returned context is a representation of *exactly this profile*
            - (left empty == not implemented in parent ProfileDensity)

        - get_generic_profile_rendering_context()

            - implemented in parent densityprofile
            - returned representation is generic for *any profile*
              (i.e. contains meta information which type is actually used)
            - passes information from get_rendering_context() through
            - returned representation is designed for easy use with templating
              engine mustache
        """
        # import here to avoid circular inclusion
        from .uniform import Uniform
        from .foil import Foil
        template_name_by_type = {
            Uniform: "uniform",
            Foil: "foil"
        }
        if self.__class__ not in template_name_by_type:
            raise RuntimeError("unkown type: {}".format(self.__class__))

        serialized_data = self.get_rendering_context()

        # create dict with all types set to false, except for the current one
        type_dict = dict(map(lambda type_name: (type_name, False),
                             template_name_by_type.values()))
        self_class_template_name = template_name_by_type[self.__class__]
        type_dict[self_class_template_name] = True

        # final context to be returned: data + type info
        returned_context = {
            "type": type_dict,
            "data": serialized_data,
        }

        # make sure it passes schema checks
        RenderedObject.check_context_for_type(DensityProfile, returned_context)

        return returned_context
