"""
This file is part of PIConGPU.
Copyright 2024 PIConGPU contributors
Authors: Brian Edward Marre
License: GPLv3+
"""

from picongpu.pypicongpu import customuserinput

import unittest


class TestCustomUserInput(unittest.TestCase):
    # test standard workflow is possible and data+tag is passed on
    def test_standard_case_works(self):
        c = customuserinput.CustomUserInput()
        data1 = {"test_data_1": 1}
        data2 = {"test_data_2": 2}

        tag1 = "tag_1"
        tag2 = "tag_2"

        c.addToCustomInput(data1, tag1)
        c.addToCustomInput(data2, tag2)

        rendering_context = c.get_rendering_context()

        self.assertEqual(rendering_context["test_data_1"], 1)
        self.assertEqual(rendering_context["test_data_2"], 2)

        tags = c.get_tags()
        self.assertIn(tag1, tags)
        self.assertIn(tag2, tags)

    def test_wrong_tags(self):
        c = customuserinput.CustomUserInput()

        data1 = {"test_data_1": 1}
        data2 = {"test_data_2": 2}

        tag1_1 = "tag_1"
        tag1_2 = "tag_1"

        # first add must succeed
        c.addToCustomInput(data1, tag1_1)
        with self.assertRaisesRegex(ValueError, "duplicate tag!"):
            c.addToCustomInput(data2, tag1_2)

        with self.assertRaisesRegex(ValueError, "tag must not be empty"):
            c.addToCustomInput(data2, "")

    def test_wrong_custom_input(self):
        c = customuserinput.CustomUserInput()

        data1_1 = {"test_data_1": 1}
        data1_2 = {"test_data_1": 2}
        empty_data = {}

        tag1 = "tag_1"
        tag2 = "tag_2"

        with self.assertRaisesRegex(ValueError, "custom input must contain at least 1 key"):
            c.addToCustomInput(empty_data, tag1)

        c.addToCustomInput(data1_1, tag1)
        with self.assertRaisesRegex(ValueError, "Key test_data_1 exist already, and specified values differ."):
            c.addToCustomInput(data1_2, tag2)

        # test same key with same value is allowed
        c.addToCustomInput(data1_1, tag2)
