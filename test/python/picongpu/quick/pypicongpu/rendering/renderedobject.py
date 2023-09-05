"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.rendering import RenderedObject

import unittest
import typeguard
from picongpu.pypicongpu.solver import YeeSolver
from picongpu.pypicongpu import Simulation
import jsonschema


class TestRenderedObject(unittest.TestCase):
    def schema_store_init(self):
        RenderedObject._schemas_loaded = False
        RenderedObject._maybe_fill_schema_store()

    def schema_store_reset(self):
        RenderedObject._schemas_loaded = False
        RenderedObject._schema_by_uri = {}

    def setUp(self):
        self.schema_store_reset()
        # if required test case can additionally init the schema store

    def test_basic(self):
        """simple example using real-world example"""
        yee = YeeSolver()
        self.assertTrue(isinstance(yee, RenderedObject))
        self.assertNotEqual(
            {}, RenderedObject._get_schema_from_class(type(yee)))
        # no throw -> schema found
        self.assertEqual(yee.get_rendering_context(),
                         yee._get_serialized())

        # manually check that schema has been loaded
        fqn = RenderedObject._get_fully_qualified_class_name(type(yee))
        uri = RenderedObject._get_schema_uri_by_fully_qualified_class_name(fqn)
        self.assertTrue(uri in RenderedObject._schema_by_uri)

    def test_not_implemented(self):
        """raises if _get_serialized() is not implemented"""
        class EmptyClass(RenderedObject):
            pass

        with self.assertRaises(NotImplementedError):
            e = EmptyClass()
            e.get_rendering_context()

    def test_no_schema(self):
        """not finding a schema raises"""
        class HasNoSchema(RenderedObject):
            def _get_serialized(self):
                return {"any": "thing"}

        with self.assertRaisesRegex(RuntimeError, ".*[Ss]chema.*"):
            h = HasNoSchema()
            h.get_rendering_context()

    def test_schema_validation_and_passthru(self):
        """schema is properly validated (and passed through)"""
        self.schema_store_init()

        class MaybeValid(RenderedObject):
            be_valid = False

            def _get_serialized(self):
                if self.be_valid:
                    return {"my_string": "ja", "num": 17}
                return {"my_string": ""}

        fqn = RenderedObject._get_fully_qualified_class_name(MaybeValid)
        uri = RenderedObject._get_schema_uri_by_fully_qualified_class_name(fqn)
        RenderedObject._schema_by_uri[uri] = {
            "properties": {
                "my_string": {"type": "string"},
                "num": {"type": "number"},
            },
            "required": ["my_string", "num"],
            "unevaluatedProperties": False,
        }

        # all okay
        maybe_valid = MaybeValid()
        maybe_valid.be_valid = True
        self.assertNotEqual({}, maybe_valid.get_rendering_context())

        maybe_valid.be_valid = False
        with self.assertRaisesRegex(Exception, ".*[Ss]chema.*"):
            maybe_valid.get_rendering_context()

    def test_invalid_schema(self):
        """schema itself is broken -> creates error"""
        self.schema_store_init()

        class HasInvalidSchema(RenderedObject):
            def _get_serialized(self):
                return {"any": "thing"}

        fqn = RenderedObject._get_fully_qualified_class_name(HasInvalidSchema)
        uri = RenderedObject._get_schema_uri_by_fully_qualified_class_name(fqn)
        # note: this is very evil injection, do not *ever* do this
        RenderedObject._schema_by_uri[uri] = {
            "type": "invalid_type_HJJE$L!BGCDHS",
        }

        h = HasInvalidSchema()
        with self.assertRaisesRegex(Exception, ".*[Ss]chema.*"):
            h.get_rendering_context()

    def test_schema_should_forbid_unevaluated_properties(self):
        """warn if schema allows unevaluated properties"""
        self.schema_store_init()

        class HasPermissiveSchema(RenderedObject):
            def _get_serialized(self):
                return {"any": "thing"}
        fqn = RenderedObject._get_fully_qualified_class_name(
            HasPermissiveSchema)
        uri = RenderedObject._get_schema_uri_by_fully_qualified_class_name(fqn)

        # schema "{}" is considered too permissive
        RenderedObject._schema_by_uri[uri] = {}

        permissive = HasPermissiveSchema()
        with self.assertLogs(level="WARNING") as caught_logs:
            # valid, but warns
            self.assertNotEqual({}, permissive.get_rendering_context())
        self.assertEqual(1, len(caught_logs.output))

    def test_fully_qualified_classname(self):
        """fully qualified classname is correctly generated"""
        # concept: define two classes of same name
        # FQN (fully qualified name) must contain their names
        # but both FQNs must be not equal

        def obj1():
            class MyClass:
                pass
            return MyClass

        def obj2():
            class MyClass:
                pass
            return MyClass

        t1 = obj1()
        t2 = obj2()
        # both are not equal
        self.assertNotEqual(t1, t2)
        # ... but type equality still works (sanity check)
        self.assertNotEqual(t1, obj1())

        fqn1 = RenderedObject._get_fully_qualified_class_name(t1)
        fqn2 = RenderedObject._get_fully_qualified_class_name(t2)

        # -> "MyClass" is contained in FQN
        self.assertTrue("MyClass" in fqn1)
        self.assertTrue("MyClass" in fqn2)
        # ... but they are not the same
        self.assertNotEqual(fqn1, fqn2)

    def test_schema_optional(self):
        """schema may define optional parameters"""
        self.schema_store_init()

        class MayReturnNone(RenderedObject):
            toreturn = None

            def _get_serialized(self):
                return {"value": self.toreturn}

        fqn = RenderedObject._get_fully_qualified_class_name(MayReturnNone)
        uri = RenderedObject._get_schema_uri_by_fully_qualified_class_name(fqn)
        RenderedObject._schema_by_uri[uri] = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {
                            "type": "null",
                        },
                        {
                            "type": "object",
                            "properties": {
                                "mandatory": {
                                    "type": "number",
                                    "exclusiveMinimum": 0,
                                    },
                                },
                            "required": ["mandatory"],
                            "unevaluatedProperties": False,
                        },
                    ],
                },
            },
            "required": ["value"],
            "unevaluatedProperties": False,
        }

        # ok:
        mrn = MayReturnNone()
        mrn.toreturn = None
        self.assertEqual({"value": None}, mrn.get_rendering_context())
        mrn.toreturn = {"mandatory": 2}
        self.assertEqual({"value": {"mandatory": 2}},
                         mrn.get_rendering_context())

        for invalid in [{"mandatory": 0}, {}, "", []]:
            with self.assertRaises(Exception):
                mrn = MayReturnNone()
                mrn.toreturn = invalid
                mrn.get_rendering_context()

    def test_check_context(self):
        """context check can be used manually"""
        yee = YeeSolver()
        context_correct = yee.get_rendering_context()
        context_incorrect = {}

        # must load schemas if required -> reset schema store
        self.schema_store_reset()
        self.assertTrue(not RenderedObject._schemas_loaded)

        # (A) context is correctly checked against the given type
        # passes:
        RenderedObject.check_context_for_type(YeeSolver, context_correct)

        # implicitly filled schema store
        self.assertTrue(RenderedObject._schemas_loaded)

        # same context is not valid for simulation object
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            RenderedObject.check_context_for_type(Simulation, context_correct)

        # incorrect context not accepted for YeeSolver
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            RenderedObject.check_context_for_type(YeeSolver, context_incorrect)

        # (B) invalid requests are rejected
        # wrong argument types
        with self.assertRaises(typeguard.TypeCheckError):
            RenderedObject.check_context_for_type("YeeSolver", context_correct)
        with self.assertRaises(typeguard.TypeCheckError):
            RenderedObject.check_context_for_type(YeeSolver, "{}")

        # types without schema
        class HasNoValidation:
            # note: don't use "Schema" to not accidentally trigger the regex
            # for the error message below
            # note: does not have to inherit from RenderedObject
            pass

        with self.assertRaisesRegex(RuntimeError, ".*[Ss]chema.*"):
            RenderedObject.check_context_for_type(HasNoValidation, {})

    def test_irregular_schema(self):
        """non-object (but valid) schemas are accepted"""
        self.schema_store_init()

        class SimpleObject(RenderedObject):
            def _get_serialized(self):
                return {}

        fqn = RenderedObject._get_fully_qualified_class_name(SimpleObject)
        uri = RenderedObject._get_schema_uri_by_fully_qualified_class_name(fqn)
        # the schema "false" is a valid schema; it rejects all inputs
        RenderedObject._schema_by_uri[uri] = False

        # there must be an error during validation & a warning issued
        with self.assertLogs(level="WARNING") as caught_logs_rejected:
            with self.assertRaises(jsonschema.exceptions.ValidationError):
                SimpleObject().get_rendering_context()
            self.assertEqual(1, len(caught_logs_rejected.output))

        # reverse: now must be accepted -- but warning still issued!
        RenderedObject._schema_by_uri[uri] = True
        with self.assertLogs(level="WARNING") as caught_logs_accepted:
            SimpleObject().get_rendering_context()
        self.assertEqual(1, len(caught_logs_accepted.output))
