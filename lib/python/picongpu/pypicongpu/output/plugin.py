"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Brian Edward Marre, Masoud Afshari
License: GPLv3+
"""

from ..util import SelfRegistering
from ..rendering import RenderedObject


import typeguard


@typeguard.typechecked
class Plugin(RenderedObject, SelfRegistering):
    """general interface for all plugins"""

    def __init__(self):
        raise NotImplementedError("abstract base class only")

    def get_rendering_context(self) -> dict:
        """
        retrieve a context valid for "any plugin"

        Problem: Every plugin has its respective schema, and it is difficult
        in JSON (particularly in a mustache-compatible way) to get the type
        of the schema.

        Solution: The normal rendering of plugins get_rendering_context()
        provides **only their parameters**, i.e. there is **no meta
        information** on types etc.

        If a generic plugin is requested one can use the schema for
        "Plugin" (this class), for which this method returns the
        correct content, which includes metainformation and the data on the
        schema itself.

        E.g.:

        .. code::

            {
                "type": {
                    "phasespace": true,
                    "auto": false,
                    ...
                },
                "data": DATA
            }

        where DATA is the serialization as returned by get_rendering_context().

        There are *two* context serialization methods for plugins:

        - get_rendering_context()

            - provided by RenderedObject parent class, serialization ("context
              building") performed by _get_serialized()
            - _get_serialized() implemented in *every plugin*
            - checks against schema of respective plugin
            - returned context is a representation of *exactly this plugin*
            - (left empty == not implemented in parent Plugin)

        - get_generic_plugin_rendering_context()

            - implemented in parent class Plugin
            - returned representation is generic for *any plugin*
              (i.e. contains meta information which type is actually used)
            - passes information from get_rendering_context() through
            - returned representation is designed for easy use with templating
              engine mustache
        """

        # final context to be returned: data + type info
        returned_context = {
            "typeID": {name: name == self._name for name in self._names},
            "data": super().get_rendering_context(),
        }

        # make sure it passes schema checks
        RenderedObject.check_context_for_type(Plugin, returned_context)

        return returned_context
