"""
This file is part of the PIConGPU.
Copyright 2021-2022 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from .. import util
from typeguard import typechecked
import typing
import jsonschema
import logging
import pathlib
import re
import json


@typechecked
class RenderedObject:
    """
    Class to be inherited from for rendering context generation

    Every object that intends to generate a rendering context (typically as a
    dict) must inherit from this class and implement _get_serialized().

    It is expected that _get_serialized() fully encodes the internal state,
    i.e. two equal (==) objects also return an equal result for
    _get_serialized().

    For external usage, the method get_rendering_context() is provided.
    It passes _get_serialized() through, checking its result against a
    predefined json schema. If this schema is not fullfilled/not available,
    an error is raised.
    """

    _schema_by_uri = util.build_typesafe_property(typing.Dict[str, dict])
    """
    store providing loaded schemas by their uri

    intended to be filled by maybe_fill_schema_store()
    """

    _schemas_loaded = False
    """
    cache if _maybe_fill_schema_store already filled the schema store
    """

    _BASE_URI = "https://registry.hzdr.de/crp/picongpu/schema/"
    """
    base URI for for schemas

    Appending a fully qualified class name to the base uri yields it full URI,
    which is used for identification purposes.
    """

    @staticmethod
    def _maybe_fill_schema_store() -> None:
        """
        fills schema_by_uri from disk if not yet done

        Does nothing if already executed previously.
        Uses static class attribute, so is global.

        Crawls predefined directory (TODO which) for ".json" files, does not
        place further restriction on naming schemes.
        """
        # if already loaded -> quit
        if RenderedObject._schemas_loaded:
            return

        # compute schema store path
        # -> find source of pypicongpu repo,
        # from there derive schema location
        here = pathlib.Path(__file__)
        schemas_path = here.parents[5] / "share/picongpu/pypicongpu/schema"
        json_fileending_re = re.compile(r"[.]json$")
        all_json_files = list(
            filter(lambda p: json_fileending_re.search(str(p)),
                   filter(lambda p: p.is_file(),
                          pathlib.Path(schemas_path).rglob("*"))))

        logging.debug("found {} schemas in {}".format(len(all_json_files),
                                                      schemas_path))

        RenderedObject._schema_by_uri = {}
        for json_file_path in all_json_files:
            with open(json_file_path, "r") as infile:
                schema = json.load(infile)
            if "$id" not in schema:
                logging.error("cant load schema, has no URI ($id) set: {}"
                              .format(json_file_path))
                continue
            uri = schema["$id"]
            if type(uri) is not str:
                raise TypeError(
                    "URI ($id) must be string: {}".format(json_file_path))

            RenderedObject._schema_by_uri[uri] = schema

        RenderedObject._schemas_loaded = True

    @staticmethod
    def _get_fully_qualified_class_name(t: type) -> str:
        """
        translate given type name to fully qualified version

        :param t: type to be resolved
        :return: pypicongpu.species.attributes.momentum.drift.Drift
        """
        return "{}.{}".format(t.__module__, t.__qualname__)

    @staticmethod
    def _get_schema_uri_by_fully_qualified_class_name(fqn: str) -> str:
        """
        get URI of json schema for given class name

        :param fqn: fully qualified class name
        :return: URI of corresponding json schema
        """
        assert RenderedObject._BASE_URI.endswith("/")
        return "{}{}".format(RenderedObject._BASE_URI, fqn)

    @staticmethod
    def _get_schema_from_class(class_type: type) -> typing.Any:
        """
        retrieve schema for given class type

        Uses get_fully_qualified_class_name() to get a class path, then derives
        URI from that and looks it up in the store.

        Note that the returned schema must not necessarily be a dict, but
        should typically be one.

        :param class_type: python type of class to look up for
        :raise RuntimeError: on schema for class not found
        :return: json schema for given class
        """
        # compute URI for lookup
        fqn = RenderedObject._get_fully_qualified_class_name(class_type)
        uri = RenderedObject._get_schema_uri_by_fully_qualified_class_name(fqn)

        # load schemas (if not done yet)
        RenderedObject._maybe_fill_schema_store()

        if uri not in RenderedObject._schema_by_uri:
            raise RuntimeError(
                "schema not found for FQN {}: URI {}".format(fqn, uri))

        schema = RenderedObject._schema_by_uri[uri]

        # validate schema
        validator = jsonschema.Draft202012Validator(schema=schema)
        validator.check_schema(schema)

        # there are schemas that are valid but not an object -> skip checks
        if type(schema) is dict:
            if "unevaluatedProperties" not in schema:
                logging.warning("schema does not explicitly forbid "
                                "unevaluated properties: {}".format(fqn))
            elif schema["unevaluatedProperties"]:
                logging.warning(
                    "schema supports unevaluated properties: {}".format(fqn))
        else:
            logging.warning("schema is not dict: {}".format(fqn))

        return schema

    def _get_serialized(self) -> dict:
        """
        return all required content for rendering as a dict
        :return: content as dictionary
        """
        raise NotImplementedError(
            "called parent _get_serialized of parent RenderedObject")

    def get_rendering_context(self) -> dict:
        """
        get rendering context representation of this object

        delegates work to _get_serialized and invokes checks performed by
        check_context_for_type().
        :raise ValidationError: on schema violiation
        :raise RuntimeError: on schema not found
        :return: self as rendering context
        """
        # to be checked against schema
        # note: load here, s.t. "not implemented error" is raised first
        serialized = self._get_serialized()
        RenderedObject.check_context_for_type(self.__class__, serialized)
        return serialized

    @staticmethod
    def check_context_for_type(type_to_check: type, context: dict) -> None:
        """
        check if the given context is valid for the given type

        Raises on error, passes silently if okay.

        :raise ValidationError: on schema violiation
        :raise RuntimeError: on schema not found
        """
        schema = RenderedObject._get_schema_from_class(type_to_check)

        fqn = RenderedObject._get_fully_qualified_class_name(type_to_check)
        uri = RenderedObject._get_schema_uri_by_fully_qualified_class_name(fqn)

        resolver = jsonschema.RefResolver(base_uri=RenderedObject._BASE_URI,
                                          referrer=uri,
                                          store=RenderedObject._schema_by_uri)
        validator = jsonschema.Draft202012Validator(schema=schema,
                                                    resolver=resolver)

        # raises on error
        validator.validate(context)
