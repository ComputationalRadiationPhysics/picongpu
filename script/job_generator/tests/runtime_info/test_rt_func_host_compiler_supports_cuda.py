# pylint: disable=missing-docstring

"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest
from typing import cast
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import alpaka_bashi.runtime_info
from alpaka_bashi.versions import get_alpaka_version_relation


class TestRtFuncHostCompilerSupportsCuda(unittest.TestCase):
    HOST_COMPILER_NVCC_VERSIONS_CASE1 = {
        GCC: [10, 11, 12, 13],
        CLANG: [15, 16, 17, 18],
        NVCC: ["12.0", "12.1", "12.2", "12.3", "12.4", "12.5", "12.6"],
    }

    CASE1_TEST_DATA_RESULTS = [
        (GCC, 10, True),
        (GCC, 13, True),
        (GCC, 9, True),
        (CLANG, 9, True),
        (CLANG, 18, True),
        (GCC, 14, False),
        (CLANG, 20, False),
        (CLANG, 99, False),
    ]

    def test_rt_func_host_compiler_supports_cuda_case1(self):
        host_compiler_support_cuda = cast(
            alpaka_bashi.runtime_info.HostCompilerSupportsCuda,
            alpaka_bashi.runtime_info.get_rt_func_host_compiler_supports_cuda(
                input_versions=self.HOST_COMPILER_NVCC_VERSIONS_CASE1,
                version_relation=get_alpaka_version_relation(),
            ),
        )
        self.assertEqual(
            host_compiler_support_cuda.get_max_version(GCC), packaging.version.parse("13")
        )
        self.assertEqual(
            host_compiler_support_cuda.get_max_version(CLANG), packaging.version.parse("18")
        )
        for compiler, version, expected_result in self.CASE1_TEST_DATA_RESULTS:
            with self.subTest(compiler=compiler, version=version, expected_result=expected_result):
                self.assertEqual(
                    host_compiler_support_cuda(compiler, packaging.version.parse(str(version))),
                    expected_result,
                )

    HOST_COMPILER_NVCC_VERSIONS_CASE2 = {
        GCC: [10, 11, 12, 13],
        CLANG: [15, 16, 17, 18],
        NVCC: ["12.0", "12.1", "12.2", "12.3"],
    }

    def test_rt_func_host_compiler_supports_cuda_case2(self):
        host_compiler_support_cuda = cast(
            alpaka_bashi.runtime_info.HostCompilerSupportsCuda,
            alpaka_bashi.runtime_info.get_rt_func_host_compiler_supports_cuda(
                input_versions=self.HOST_COMPILER_NVCC_VERSIONS_CASE2,
                version_relation=get_alpaka_version_relation(),
            ),
        )
        self.assertEqual(
            host_compiler_support_cuda.get_max_version(GCC), packaging.version.parse("12")
        )
        self.assertEqual(
            host_compiler_support_cuda.get_max_version(CLANG), packaging.version.parse("16")
        )
