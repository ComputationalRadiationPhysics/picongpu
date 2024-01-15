"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

import typing
from typeguard import typechecked
import math
import datetime
import sympy
import chevron
import re
import logging
import pathlib
import functools


@typechecked
class Renderer:
    """
    helper class to render Mustache templates

    Collection of (static) functions to render Mustache templates.

    Also contains checks for structure of passed context (JSON) objects and the
    JSON schema checks and related functions (loading schema database, checking
    schemas, looking up schemas by type etc.)
    """

    @staticmethod
    def __check_rendering_context_recursive(path: str, context: dict) -> None:
        """
        Implement check_rendering_context

        If checks succeed passes silently, else raises.

        :param path: description of current location in dictionary
        :param context: context to be checked
        """
        for key, value in context.items():
            # note on typechecks here: we want *exact* types,
            # so don't use instanceof()
            if str != type(key):
                raise TypeError("all keys must be strings: {}".format(path))
            if "." in key:
                raise ValueError("keys may NOT contain dot: {}.{}".format(path, key))
            if key.startswith("_"):
                raise ValueError("keys may NOT start with underscore: {}.{}".format(path, key))

            # value validation
            if type(value) is dict:
                if {} == value:
                    raise TypeError("leaf must not be empty dict")
                # dict -> recursive call
                Renderer.__check_rendering_context_recursive("{}.{}".format(path, key), value)
            elif type(value) is list:
                # may only contain dicts
                # note: this is not a strict mustache requirement, but only
                # exists to prevent developer-screwups (in mustache, rendering
                # mylist: [1, 2, 3] is performed by
                # {{#mylist}}{{{.}}}{{/mylist}}, which is somewhat unintuitive)
                not_dict = list(filter(lambda e: type(e) is not dict, value))
                if 0 != len(not_dict):
                    raise TypeError("lists may only contains dicts: {}.{}".format(path, key))
                # check the children
                for i in range(len(value)):
                    Renderer.__check_rendering_context_recursive("{}[{}]".format(path, i), value[i])
            else:
                # leaf
                invalid_floats = [math.inf, -math.inf, math.nan]
                if value in invalid_floats:
                    raise ValueError("invalid value for leaf: {} at {}.{}".format(value, path, key))

                allowed_types = [str, bool, type(None), int, float]
                if type(value) not in allowed_types:
                    raise TypeError(
                        "leaf may only be str, bool, None, number;" " found: {} at {}.{}".format(type(value), path, key)
                    )

    @staticmethod
    def check_rendering_context(context: typing.Any) -> None:
        """
        check if a context object may be renderd

        If checks succeed passes silently, else raises.

        Must be used *before* the preprocessor (as it checks that keys reserved
        for the preprocessor are not yet used).

        Performs if the given object is acceptable as rendering context:
        - is dict
        - leafs are string, boolean, None, int, or float (or empty list)
        - child nodes are leaf, set or list
        - list items must be dict
        - keys are strings
        - keys do *not* contain dot (.)
        - keys do *not* begin with underscore (_) -> reserved for preprocessor

        :param context: object to be checked
        :raises Exception: on check failure
        """
        if dict != type(context):
            raise TypeError("rendering context must be dict")
        Renderer.__check_rendering_context_recursive("ROOT", context)

    @staticmethod
    def __get_context_preprocessed_recursive(context: dict) -> dict:
        """
        implements get_context_preprocessed

        operates recursively

        :param context: dictionary to be preprocessed
        """
        pp = {}
        for key, value in context.items():
            if type(value) is dict:
                # dict -> decent
                pp[key] = Renderer.__get_context_preprocessed_recursive(value)
            elif type(value) is list:
                # list: add _last, _first
                new_list = []
                for i in range(len(value)):
                    elem = Renderer.__get_context_preprocessed_recursive(value[i])
                    elem["_first"] = 0 == i
                    elem["_last"] = len(value) - 1 == i
                    new_list.append(elem)
                pp[key] = new_list
            elif type(value) in [int, float]:
                # translate numbers to C++-compatible string
                sympy_num = sympy.sympify(value)
                pp[key] = sympy.printing.ccode(sympy_num)
            else:
                # passthru
                pp[key] = value

        return pp

    @staticmethod
    def get_context_preprocessed(context: dict) -> dict:
        """
        preprocess a context object

        Applies the following preproccessing to an object:
        - list items have property "_first" and "_last" (boolean) added
        - numbers are translated to C++-compatible strings

        rejects unchecked rendering contexts

        :param context: context to be preprocessed
        :return: preprocessed copy of context dict
        """
        # delegate functionality to recursive function
        pp = Renderer.__get_context_preprocessed_recursive(context)

        # add properties at top level
        pp["_date"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        return pp

    @staticmethod
    def get_rendered_template(context: dict, template: str) -> str:
        """
        get rendered version of the provided string

        Renders a given string using the given context object.

        :param context: (checked and preprocessed) rendering context
        :param template: string containing template to be rendered
        :return: rendered template
        """
        # warns for unkown variables
        # matches blocks from at least two {{ to at least two }}
        # -> does not actually count braces (i.e. is strictly incorrect)
        mustache_block_re = re.compile(r"{{({*(?:}?[^}]+)*}*)}}")
        for match in mustache_block_re.finditer(template):
            block_content = match.group(1)
            if "" == block_content:
                logging.warning("empty mustache block encountered")
            if block_content[0] not in "{^#/>!":
                # note: use string composition instead of normal formatstrings
                logging.warning(
                    "do NOT use HTML escaped syntax (only {{two braces}}) for " "vars, offending var: " + match.group(1)
                )
        return chevron.render(template, context, warn=True)

    @staticmethod
    def render_directory(context: dict, path: str) -> None:
        """
        Render all templates inside a given directory and remove the templates

        Recursively find all files inside given path and render the files
        ending in ".mustache" to the same name without the ending.
        The original ".mustache" files are renamed with a dot "." prefix.

        :param context: (checked and preprocessed) rendering context
        :param path: directory containing ".mustache" files
        """
        if not pathlib.Path(path).is_dir():
            raise ValueError("is not a directory: {}".format(path))

        mustache_fileending_re = re.compile(r"[.]mustache$")
        all_mustache_files = list(
            filter(
                lambda p: mustache_fileending_re.search(str(p)),
                filter(lambda p: p.is_file(), pathlib.Path(path).rglob("*")),
            )
        )
        for template_path in all_mustache_files:
            rendered_path = pathlib.Path(mustache_fileending_re.sub("", str(template_path)))
            if rendered_path.exists():
                raise ValueError("would overwrite {}, aborting".format(rendered_path))

            with open(rendered_path, "w") as outfile:
                with open(template_path, "r") as infile:
                    template_str = infile.read()
                    rendered = Renderer.get_rendered_template(context, template_str)
                    outfile.write(rendered)

            # prefix filename with .
            # (on that note: screw pathlib for only disassembling, but not
            # reassembling paths from parts)
            parts = list(template_path.parts)
            parts[-1] = "." + parts[-1]
            new_path = functools.reduce(lambda a, b: a / b, map(lambda s: pathlib.Path(s), parts))

            template_path.rename(new_path)
