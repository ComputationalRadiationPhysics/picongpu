# pylint: disable=missing-docstring

"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import alpaka_bashi.alpaka_filter
import packaging.version

TEST_CUDA_VER = packaging.version.parse("12.4")


class TestOnlyCUDABackends(unittest.TestCase):
    BACKEND_COMBINATIONS_VALID_CASE = [
        [bashi.CompilerBackendCombination(GCC, NVCC, [ALPAKA_ACC_GPU_CUDA_ENABLE])],
        [bashi.CompilerBackendCombination(ICPX, HIPCC, [ALPAKA_ACC_GPU_CUDA_ENABLE])],
        [
            bashi.CompilerBackendCombination(
                GCC,
                NVCC,
                [
                    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                    ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE,
                    ALPAKA_ACC_GPU_CUDA_ENABLE,
                ],
            )
        ],
        [
            bashi.CompilerBackendCombination(
                CLANG,
                NVCC,
                [
                    ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE,
                    ALPAKA_ACC_GPU_CUDA_ENABLE,
                ],
            ),
            bashi.CompilerBackendCombination(
                CLANG_CUDA,
                CLANG_CUDA,
                [
                    ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE,
                    ALPAKA_ACC_GPU_CUDA_ENABLE,
                ],
            ),
        ],
    ]

    def test_only_cuda_compiler_backends_valid(self):
        for backend_list in self.BACKEND_COMBINATIONS_VALID_CASE:
            with self.subTest(backend_list=backend_list):
                self.assertTrue(
                    alpaka_bashi.alpaka_filter.only_cuda_compiler_backends(backend_list),
                    backend_list,
                )

    BACKEND_COMBINATIONS_INVALID_CASE = [
        [
            bashi.CompilerBackendCombination(
                CLANG_CUDA, CLANG_CUDA, [ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE]
            )
        ],
        [
            bashi.CompilerBackendCombination(
                CLANG_CUDA, CLANG_CUDA, [ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE]
            ),
            bashi.CompilerBackendCombination(
                ICPX, ICPX, [ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ALPAKA_ACC_ONEAPI_FPGA_ENABLE]
            ),
        ],
        [
            bashi.CompilerBackendCombination(
                CLANG_CUDA,
                CLANG_CUDA,
                [ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ALPAKA_ACC_GPU_CUDA_ENABLE],
            ),
            bashi.CompilerBackendCombination(
                ICPX, ICPX, [ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ALPAKA_ACC_ONEAPI_FPGA_ENABLE]
            ),
            bashi.CompilerBackendCombination(
                GCC,
                NVCC,
                [ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ALPAKA_ACC_GPU_CUDA_ENABLE],
            ),
        ],
    ]

    def test_only_cuda_compiler_backends_invalid(self):
        for backend_list in self.BACKEND_COMBINATIONS_INVALID_CASE:
            with self.subTest(backend_list=backend_list):
                self.assertFalse(
                    alpaka_bashi.alpaka_filter.only_cuda_compiler_backends(backend_list),
                    backend_list,
                )
