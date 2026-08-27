# pylint: disable=missing-docstring

"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0
"""

from typing import Dict, List
import unittest
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
import alpaka_bashi.combination_modifier.execution_type
from utils import parse_param_value_tuples


class TestFindLatestCudaSdkMinorVersions(unittest.TestCase):
    COMBINATION_LIST = [
        parse_param_value_tuples(row)
        for row in (
            [
                (HOST_COMPILER, CLANG, 18),
                (DEVICE_COMPILER, NVCC, 12.4),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4),
            ],
            [
                (HOST_COMPILER, GCC, 12),
                (DEVICE_COMPILER, NVCC, 12.1),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1),
            ],
            [
                (HOST_COMPILER, GCC, 8),
                (DEVICE_COMPILER, NVCC, 12.1),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1),
            ],
            [
                (HOST_COMPILER, CLANG, 16),
                (DEVICE_COMPILER, NVCC, 12.2),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2),
            ],
            [
                (HOST_COMPILER, CLANG, 17),
                (DEVICE_COMPILER, NVCC, 12.1),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.1),
            ],
            [
                (HOST_COMPILER, GCC, 12),
                (DEVICE_COMPILER, NVCC, 12.8),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.8),
            ],
            [
                (HOST_COMPILER, CLANG, 17),
                (DEVICE_COMPILER, NVCC, 11.1),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.1),
            ],
            [
                (HOST_COMPILER, CLANG, 17),
                (DEVICE_COMPILER, NVCC, 11.8),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.8),
            ],
            [
                (HOST_COMPILER, CLANG, 17),
                (DEVICE_COMPILER, NVCC, 11.6),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 11.6),
            ],
            [
                (HOST_COMPILER, CLANG_CUDA, 18),
                (DEVICE_COMPILER, CLANG_CUDA, 18),
                (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.2),
            ],
        )
    ]

    def test_find_latest_cuda_sdk_minor_versions(self) -> None:
        expected_result: Dict[ValueName, List[ValueVersion]] = {
            GCC: sorted([packaging.version.parse(str(ver)) for ver in [12.1, 12.8]]),
            CLANG: sorted([packaging.version.parse(str(ver)) for ver in [11.1, 11.8, 12.1, 12.4]]),
            CLANG_CUDA: sorted([packaging.version.parse(str(ver)) for ver in [12.2]]),
        }

        result = (
            alpaka_bashi.combination_modifier.execution_type.find_latest_cuda_sdk_minor_versions(
                self.COMBINATION_LIST
            )
        )

        for compiler in (GCC, CLANG, CLANG_CUDA):
            with self.subTest(compiler=compiler):
                sorted_result = sorted(result[compiler])
                self.assertEqual(
                    sorted_result,
                    expected_result[compiler],
                    f"Compiler {compiler}: {sorted_result} != {expected_result[compiler]}",
                )
