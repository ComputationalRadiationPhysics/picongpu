# SPDX-FileCopyrightText: PIConGPU contributors
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This file is part of PIConGPU.
Copyright 2021-2025 PIConGPU contributors
Authors: Hannes Troepgen, Brian Edward Marre, Julian Lenz
License: GPLv3+
"""

from unittest import TestCase
from pathlib import Path

import pytest

from picongpu.picmi.simulation import _normalise_template_dir


class TestNormaliseTemplateDir(TestCase):
    def test_single_string(self):
        existing_dir_string = "."
        assert list(_normalise_template_dir(existing_dir_string)) == [Path(existing_dir_string)]

    def test_none(self):
        assert list(_normalise_template_dir(None)) == []

    def test_path(self):
        existing_dir = Path()
        assert list(_normalise_template_dir(existing_dir)) == [existing_dir]

    def test_mixed_iterable(self):
        mixed_iter = [".", Path(), None]
        assert list(_normalise_template_dir(mixed_iter)) == [Path(), Path()]

    def test_disallows_non_existent_paths(self):
        non_existent_dir = Path("non_existent_dir").absolute()
        if non_existent_dir.exists():
            raise ValueError(f"Test could not proceed because {non_existent_dir=} does exist.")
        with pytest.raises(ValueError, match=".*is not an existing directory.*"):
            _normalise_template_dir(non_existent_dir)
