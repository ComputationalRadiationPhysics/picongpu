"""
This file is part of PIConGPU.
Copyright 2021-2024 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import typeguard
import typing
import jsonschema
import referencing
import logging
import pathlib
import re
import json


@typeguard.typechecked
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

    _registry: referencing.Registry = referencing.Registry()
    """
    store providing all found schemas

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

    def __hash__(self):
        """custom hash function for indexing in dicts"""
        hash_value = hash(type(self))

        for value in self.__dict__.values():
            try:
                if value is not None:
                    hash_value += hash(value)
            except TypeError:
                print(self)
                print(type(self))
                raise TypeError
        return hash_value

    @staticmethod
    def _maybe_fill_schema_store() -> None:
        """
        fills schema_by_uri from disk if not yet done

        Does nothing if already executed previously.
        Uses static class attribute, so is global.

        Crawls predefined directory "share/picongpu/pypicongpu/schema" for ".json" files,
        otherwise no restriction on naming of schemas.
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
            filter(
                lambda p: json_fileending_re.search(str(p)),
                filter(lambda p: p.is_file(), pathlib.Path(schemas_path).rglob("*")),
            )
        )

        logging.debug("found {} schemas in {}".format(len(all_json_files), schemas_path))

        for json_file_path in all_json_files:
            with open(json_file_path, "r") as infile:
                schema = json.load(infile)
            if "$id" not in schema:
                logging.error("cant load schema, has no URI ($id) set: {}".format(json_file_path))
                continue
            uri = schema["$id"]
            if type(uri) is not str:
                raise TypeError("URI ($id) must be string: {}".format(json_file_path))

            resource = referencing.Resource(contents=schema, specification=referencing.jsonschema.DRAFT202012)

            # registries are immutable, every call will return new instance and leave old instance unchanged
            RenderedObject._registry = RenderedObject._registry.with_resource(uri, resource)

        # crawl all added resources
        RenderedObject._registry = RenderedObject._registry.crawl()

        # mark registry as loaded
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

        try:
            schema = RenderedObject._registry.contents(uri)
        except referencing.exceptions.NoSuchResource:
            raise referencing.exceptions.NoSuchResource("schema not found for FQN {}: URI {}".format(fqn, uri))

        # validate schema
        validator = jsonschema.Draft202012Validator(schema=schema)
        validator.check_schema(schema)

        # there are schemas that are valid but not an object -> skip checks
        if type(schema) is dict:
            if "unevaluatedProperties" not in schema:
                logging.warning("schema does not explicitly forbid " "unevaluated properties: {}".format(fqn))
            # special exemption for custom user input which is never evaluated
            elif schema["unevaluatedProperties"] and fqn != "picongpu.pypicongpu.customuserinput.CustomUserInput":
                logging.warning("schema supports unevaluated properties: {}".format(fqn))
        else:
            logging.warning("schema is not dict: {}".format(fqn))

        return schema

    def _get_serialized(self) -> dict | None:
        """
        return all required content for rendering as a dict
        :return: content as dictionary
        """
        raise NotImplementedError("called parent _get_serialized of parent RenderedObject")

    def get_rendering_context(self) -> dict | None:
        """
        get rendering context representation of this object

        delegates work to _get_serialized and invokes checks performed by
        check_context_for_type().
        :raise ValidationError: on schema violation
        :raise RuntimeError: on schema not found
        :return: self as rendering context
        """
        # to be checked against schema
        # note: load here, s.t. "not implemented error" is raised first
        serialized = self._get_serialized()
        RenderedObject.check_context_for_type(self.__class__, serialized)
        return serialized

    @staticmethod
    def check_context_for_type(type_to_check: type, context: dict | None) -> None:
        """
        check if the given context is valid for the given type

        Raises on error, passes silently if okay.

        :raise ValidationError: on schema violation
        :raise RuntimeError: on schema not found
        """

        schema = RenderedObject._get_schema_from_class(type_to_check)
        validator = jsonschema.Draft202012Validator(schema=schema, registry=RenderedObject._registry)

        # raises on error
        validator.validate(context)
