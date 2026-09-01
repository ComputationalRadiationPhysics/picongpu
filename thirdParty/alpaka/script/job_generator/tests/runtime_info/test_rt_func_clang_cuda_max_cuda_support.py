# pylint: disable=missing-docstring

"""Copyright 2026 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest
import packaging
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.version.dependencies.clang_cuda import ClangCudaSDKSupport
from alpaka_bashi.runtime_info import ClangCUDAMaxSupportsCuda


class TestRtClangCUDAMaxSupportsCuda(unittest.TestCase):
    CLANG_CUDA_VERSIONS_CASE1 = {CLANG_CUDA: [14, 15, 16, 17, 18, 19]}
    CLANG_CUDA_SDK_SUPPORT_CASE1 = [
        ClangCudaSDKSupport("7", "9.2"),
        ClangCudaSDKSupport("8", "10.0"),
        ClangCudaSDKSupport("10", "10.1"),
        ClangCudaSDKSupport("12", "11.0"),
        ClangCudaSDKSupport("13", "11.2"),
        ClangCudaSDKSupport("14", "11.5"),
        ClangCudaSDKSupport("16", "11.8"),
        ClangCudaSDKSupport("17", "12.1"),
        ClangCudaSDKSupport("18", "12.3"),
        ClangCudaSDKSupport("22", "13.0"),
    ]

    def test_rt_clang_cuda_max_cuda_support_case1(self):
        rt = ClangCUDAMaxSupportsCuda(
            bashi.VersionRelation(clang_cuda_max_cuda_version=self.CLANG_CUDA_SDK_SUPPORT_CASE1),
            self.CLANG_CUDA_VERSIONS_CASE1,
        )
        self.assertEqual(rt.max_cuda_sdk_version, packaging.version.parse("12.3"))
        self.assertTrue(rt(packaging.version.parse("12.0")))
        self.assertTrue(rt(packaging.version.parse("12.3")))
        self.assertFalse(rt(packaging.version.parse("12.4")))
        self.assertFalse(rt(packaging.version.parse("13.0")))

    CLANG_CUDA_VERSIONS_CASE2 = {CLANG_CUDA: [14, 15, 16, 17, 18]}
    CLANG_CUDA_SDK_SUPPORT_CASE2 = [
        ClangCudaSDKSupport("7", "9.2"),
        ClangCudaSDKSupport("8", "10.0"),
        ClangCudaSDKSupport("10", "10.1"),
        ClangCudaSDKSupport("12", "11.0"),
        ClangCudaSDKSupport("13", "11.2"),
        ClangCudaSDKSupport("14", "11.5"),
        ClangCudaSDKSupport("16", "11.8"),
        ClangCudaSDKSupport("17", "12.1"),
        ClangCudaSDKSupport("18", "12.3"),
        ClangCudaSDKSupport("22", "13.0"),
    ]

    def test_rt_clang_cuda_max_cuda_support_case2(self):
        rt = ClangCUDAMaxSupportsCuda(
            bashi.VersionRelation(clang_cuda_max_cuda_version=self.CLANG_CUDA_SDK_SUPPORT_CASE2),
            self.CLANG_CUDA_VERSIONS_CASE2,
        )
        self.assertEqual(rt.max_cuda_sdk_version, packaging.version.parse("12.3"))
        self.assertTrue(rt(packaging.version.parse("12.0")))
        self.assertTrue(rt(packaging.version.parse("12.3")))
        self.assertFalse(rt(packaging.version.parse("12.4")))
        self.assertFalse(rt(packaging.version.parse("13.0")))

    CLANG_CUDA_VERSIONS_CASE3 = {CLANG_CUDA: [14, 15, 16, 17, 18, 23]}
    CLANG_CUDA_SDK_SUPPORT_CASE3 = [
        ClangCudaSDKSupport("7", "9.2"),
        ClangCudaSDKSupport("8", "10.0"),
        ClangCudaSDKSupport("10", "10.1"),
        ClangCudaSDKSupport("12", "11.0"),
        ClangCudaSDKSupport("13", "11.2"),
        ClangCudaSDKSupport("14", "11.5"),
        ClangCudaSDKSupport("16", "11.8"),
        ClangCudaSDKSupport("17", "12.1"),
        ClangCudaSDKSupport("18", "12.3"),
        ClangCudaSDKSupport("22", "13.0"),
    ]

    def test_rt_clang_cuda_max_cuda_support_case3(self):
        rt = ClangCUDAMaxSupportsCuda(
            bashi.VersionRelation(clang_cuda_max_cuda_version=self.CLANG_CUDA_SDK_SUPPORT_CASE3),
            self.CLANG_CUDA_VERSIONS_CASE3,
        )
        self.assertEqual(rt.max_cuda_sdk_version, packaging.version.parse("13.0"))
        self.assertTrue(rt(packaging.version.parse("12.0")))
        self.assertTrue(rt(packaging.version.parse("12.3")))
        self.assertTrue(rt(packaging.version.parse("12.4")))
        self.assertTrue(rt(packaging.version.parse("13.0")))
        self.assertFalse(rt(packaging.version.parse("13.1")))
