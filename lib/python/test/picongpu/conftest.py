"""
Pytest configuration for PIConGPU test suite.

This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
License: GPLv3+
"""

import pytest


def pytest_collection_modifyitems(items):
    """
    Automatically apply markers based on test file location.

    - Tests in compiling/ get @pytest.mark.slow
    - Tests in end_to_end/ get @pytest.mark.slow
    - Tests in quick/ remain unmarked (fast)
    """
    for item in items:
        nodeid = item.nodeid

        if "/compiling/" in nodeid:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.compiling)

        if "/end_to_end/" in nodeid:
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.end_to_end)
