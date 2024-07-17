"""
This file is part of PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.species.operation.densityprofile import DensityProfile

import unittest

from picongpu.pypicongpu.species.operation.densityprofile import Uniform
from picongpu.pypicongpu.rendering import RenderedObject

import referencing


class TestDensityProfile(unittest.TestCase):
    def tearDown(self):
        # rendering with type test tampers with the schema store:
        # reset it after tests
        RenderedObject._schemas_loaded = False
        RenderedObject._schema_by_uri = {}

    class DummyCheckNotImplemented(DensityProfile):
        def __init__(self):
            pass

    def test_abstract(self):
        """density profile is abstract"""
        with self.assertRaises(NotImplementedError):
            DensityProfile()

        # also, check() is not implemented
        dummy = self.DummyCheckNotImplemented()
        with self.assertRaises(NotImplementedError):
            dummy.check()

    def test_rendering_not_implemented(self):
        """rendering method is defined, but not implemented"""
        dummy = self.DummyCheckNotImplemented()

        with self.assertRaises(NotImplementedError):
            dummy.get_rendering_context()

    def test_rendering_with_type(self):
        """object with added type information is returned & validated"""
        # note: use valid objects here b/c the schema enforces non-dummy types
        uniform = Uniform()
        uniform.density_si = 1

        # schemas must be loaded by context request
        RenderedObject._schemas_loaded = False
        RenderedObject._registry = referencing.Registry()

        context = uniform.get_generic_profile_rendering_context()

        # schemas now loaded
        self.assertTrue(RenderedObject._schemas_loaded)

        self.assertEqual(context["data"], uniform.get_rendering_context())

        # contains information on all types
        self.assertEqual(context["type"], {"uniform": True, "foil": False, "gaussian": False})

        # is actually validated against "DensityProfile" schema
        RenderedObject._schemas_loaded = False
        schema = RenderedObject._get_schema_from_class(DensityProfile)

        # check 1: schema actually enforce existance of all keys
        self.assertTrue("uniform" in schema["properties"]["type"]["required"])
        # TODO: copy line above for more types

        # check 1b: all keys that are available for the "type" dict are
        # required
        self.assertEqual(
            set(schema["properties"]["type"]["required"]),
            set(schema["properties"]["type"]["properties"].keys()),
        )

        # check 2: break the schema, schema rejects everything now
        RenderedObject._registry = referencing.Registry()
        with self.assertRaises(referencing.exceptions.NoSuchResource):
            # schema now rejects everything
            # -> must also reject previously correct context
            uniform.get_generic_profile_rendering_context()
