"""
This file is part of the PIConGPU.
Copyright 2021-2023 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu.rendering import Renderer

import unittest
import math
import logging
import os
import tempfile
import pathlib


class TestRenderer(unittest.TestCase):
    def test_check_rendering_context_basic(self):
        """sanity-check valid context"""
        # no raise
        Renderer.check_rendering_context({})
        Renderer.check_rendering_context({
            "key": 1,
            "ya": None,
            "bool": True,
            "grades": [
                {"grade": 1},
                {"grade": 2},
                {"grade": 1},
                {"grade": 1},
            ],
            "object": {
                "nested": {
                    "num": 3.5,
                },
            },
        })

    def test_check_rendering_context_type(self):
        """only dicts may be rendering context"""
        for invalid in [None, "asd", 123, ["123"]]:
            with self.assertRaises(TypeError):
                Renderer.check_rendering_context(invalid)

    def test_check_rendering_context_invalid_composition(self):
        """rendering context may not be constructed arbitrarily"""
        # wrong leaves
        for leaf in [set(), ({"key": 1},)]:
            with self.assertRaisesRegex(TypeError, ".*[Ll]eaf.*"):
                Renderer.check_rendering_context({"key": leaf})

        # list items not dict
        with self.assertRaisesRegex(TypeError, ".*[Ll]ist.*"):
            Renderer.check_rendering_context({
                "list": [1, 2, 3, None],
            })

        # tuples not accepted as list replacement
        with self.assertRaises(TypeError):
            Renderer.check_rendering_context({
                "not_list": tuple({"k": "v"}),
            })

    def test_check_rendering_context_special_values(self):
        """special values are correctly allowed/disallowed"""
        for valid_leaf in ["", 0, -1.2, [], None, True, False]:
            Renderer.check_rendering_context({"key": valid_leaf})

        for invalid_leaf in [math.nan, math.inf, -math.inf]:
            with self.assertRaisesRegex(ValueError, ".*[Ll]eaf.*"):
                Renderer.check_rendering_context({"key": invalid_leaf})

        # empty dict must be *type* error (because invalid type for leaf)
        with self.assertRaisesRegex(TypeError, ".*[Ll]eaf.*"):
            Renderer.check_rendering_context({"key": dict()})

    def test_check_rendering_context_invalid_keys(self):
        """keys must be well-formed"""
        # keys must be string
        with self.assertRaisesRegex(TypeError, ".*[Ss]tring.*"):
            Renderer.check_rendering_context({123: "jaja"})

        # keys may not begin with _
        with self.assertRaisesRegex(ValueError, ".*[Uu]nderscore.*"):
            Renderer.check_rendering_context({"_reserved": "x"})

        # keys may not contain .
        for invalid_key in [".asd", "hjasd.asd", "asd.", "."]:
            with self.assertRaisesRegex(ValueError, ".*[Dd]ot.*"):
                Renderer.check_rendering_context({invalid_key: "x"})

    def test_context_preprocessor_conversion(self):
        """leafs are translated to string if applicable"""
        leaf_translations = [
            (None, None),
            ("any_string", "any_string"),
            (True, True),
            ([], []),
            (0, "0"),
            (-1.5, "-1.5"),
        ]

        for original, expected_translation in leaf_translations:
            orig_context = {
                "key": original,
            }
            translated_context = \
                Renderer.get_context_preprocessed(orig_context)
            self.assertEqual(translated_context["key"], expected_translation)
            # orig_context not touched
            self.assertEqual(orig_context["key"], original)

        # precision kept (double)
        # preliminaries: check python
        # this particular test number is chosen s.t. all *16* digits are
        # preserved, just by pure conincidence (actually only 15 digits are)
        double_num_str = "9785135.876452025"
        # check that python preserves this precision
        self.assertEqual(double_num_str, str(float(double_num_str)))
        context_orig = {
            "num": float(double_num_str),
        }
        context_pp = Renderer.get_context_preprocessed(context_orig)
        self.assertEqual(context_pp["num"], double_num_str)

        # precision kept (64 bit int, typical value of size_t)
        max_uint_str = "18446744073709551616"
        context_orig = {
            "num": int(max_uint_str),
        }
        context_pp = Renderer.get_context_preprocessed(context_orig)
        self.assertEqual(context_pp["num"], max_uint_str)

    def test_context_preprocessor_add_list_attrs(self):
        """preprocessing adds appropriate attributes"""
        context_orig = {
            "l": [
                {"val": 0},
                {"val": 1},
                {"val": 2},
                {"val": 3},
                {"val": 4},
                {"val": 5},
            ]
        }
        context_pp = Renderer.get_context_preprocessed(context_orig)
        self.assertEqual(context_pp["l"][0]["_last"], False)
        self.assertEqual(context_pp["l"][0]["_first"], True)

        for i in range(1, 5):
            self.assertEqual(context_pp["l"][i]["_last"], False)
            self.assertEqual(context_pp["l"][i]["_first"], False)

        self.assertEqual(context_pp["l"][5]["_last"], True)
        self.assertEqual(context_pp["l"][5]["_first"], False)

    def test_context_preprocessor_add_top_level_attrs(self):
        """highest level dict gets special attrs"""
        context_orig = {
            "child": {
                "val": "x",
            },
        }
        context_pp = Renderer.get_context_preprocessed(context_orig)

        self.assertEqual(context_orig["child"], context_pp["child"])
        self.assertTrue("_date" in context_pp)

    def test_get_rendered_template(self):
        """sanity-check the rendering engine"""
        self.assertEqual("", Renderer.get_rendered_template({}, ""))
        self.assertEqual("", Renderer.get_rendered_template(
            {"yes": False},
            "{{#yes}}xxx{{/yes}}"))
        self.assertEqual("xyz", Renderer.get_rendered_template(
            {"alphabet": [
                {"letter": "x"},
                {"letter": "y"},
                {"letter": "z"},
            ]},
            "{{#alphabet}}{{{letter}}}{{/alphabet}}"))
        self.assertEqual("subsetvalue", Renderer.get_rendered_template(
            {
                "sub": {
                    "set": {
                        "value": "subsetvalue",
                    },
                 },
            },
            "{{{sub.set.value}}}"))

    def test_get_rendered_template_warn_escape(self):
        """if HTML-escaping is used, a warning is issued"""
        warn_cnt_by_template = {
            "{{#if}}x{{/if}}": 0,
            "{{var}}": 1,
            "{{var}}{{num}}": 2,
            "{{{var}}}": 0,
            "{{! comment only }}": 0,
        }

        default_context = {
            "if": True,
            "var": "asd",
            "num": "17",
        }

        for template, expected_warn_num in warn_cnt_by_template.items():
            with self.assertLogs(level="WARNING") as caught_logs:
                # at least one warning is required or this will fails
                logging.warning("workaround warning")
                Renderer.get_rendered_template(default_context, template)
            # -> subtract workaround warning
            self.assertEqual(expected_warn_num, len(caught_logs.output) - 1)

    def test_render_directory(self):
        """rendering directory traverses correct files"""
        tmpdir = None
        # get temp dir
        with tempfile.TemporaryDirectory() as tmpdir_path:
            tmpdir = tmpdir_path
            os.chdir(tmpdir)

            os.mkdir("sub")
            os.mkdir("sub/dir")

            content_by_path = {
                "not_tpl": "{{{var}}}",
                "var.mustache": "{{{var}}}",
                "empty.mustache": "",
                "sub/dir/file": "1",
                "sub/dir/num.mustache": "{{{num}}}",
            }

            for path, content in content_by_path.items():
                with open(path, "w") as file:
                    file.write(content)

            default_context = {
                "var": "a string",
                "num": "0",
            }

            expected_content_by_path = {
                "not_tpl": "{{{var}}}",
                "var": "a string",
                "empty": "",
                "sub/dir/file": "1",
                "sub/dir/num": "0",
            }
            Renderer.render_directory(default_context, tmpdir)

            # all files exist & content is correct
            for path, expected_content in expected_content_by_path.items():
                with open(path, "r") as file:
                    content = file.read()
                    self.assertEqual(content, expected_content)

            # no other files have been created
            # and all mustache-files are prefixed with a .
            expected_files = {
                "not_tpl",
                ".var.mustache",
                "var",
                ".empty.mustache",
                "empty",
                "sub/dir/file",
                "sub/dir/.num.mustache",
                "sub/dir/num",
            }
            existing_files = set(
                map(lambda p: str(p.relative_to(tmpdir)),
                    filter(lambda p: p.is_file(),
                           pathlib.Path(tmpdir).rglob("*"))))
            self.assertEqual(existing_files, expected_files)

        # temp dir now deleted (b/c left "with" block)
        # -> raises on non-existing dir
        self.assertTrue(not os.path.exists(tmpdir))
        with self.assertRaises(ValueError):
            Renderer.render_directory({}, tmpdir)

    def test_render_directory_overwrite(self):
        """reject if files would be overwritten"""
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            # create files
            for fname in ["file", "file.mustache"]:
                with open(fname, "w"):
                    pass

            with self.assertRaisesRegex(ValueError, ".*over.*"):
                Renderer.render_directory({}, tmpdir)
