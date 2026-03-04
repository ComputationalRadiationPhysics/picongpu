"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

import unittest
from pathlib import Path

from picongpu.picmi.simulation import _normalise_template_dir


class TestNormaliseTemplateDir(unittest.TestCase):
    def test_single_string(self):
        existing_dir_string = "."
        self.assertSequenceEqual(_normalise_template_dir(existing_dir_string), (Path(existing_dir_string),))

    def test_none(self):
        self.assertSequenceEqual(_normalise_template_dir(None), tuple())

    def test_path(self):
        existing_dir = Path()
        self.assertSequenceEqual(_normalise_template_dir(existing_dir), (existing_dir,))

    def test_mixed_iterable(self):
        mixed_iter = [".", Path(), None]
        self.assertSequenceEqual(_normalise_template_dir(mixed_iter), (Path(), Path()))

    def test_disallows_non_existent_paths(self):
        non_existent_dir = Path("non_existent_dir").absolute()
        if non_existent_dir.exists():
            # If this ever happens, we must come up with a more robust way of handling this.
            raise ValueError(f"Test could not proceed because {non_existent_dir=} does exist.")
        with self.assertRaisesRegex(ValueError, ".*is not an existing directory.*"):
            _normalise_template_dir(non_existent_dir)
