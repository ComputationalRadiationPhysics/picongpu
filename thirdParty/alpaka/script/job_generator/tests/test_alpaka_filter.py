# pylint: disable=missing-docstring

"""Copyright 2025 Simeon Ehrig
SPDX-License-Identifier: MPL-2.0

Custom filter for alpaka specific filter rules.
"""

import unittest
import io
import itertools
from typing import Dict, Callable, cast
import bashi
from bashi.globals import *  # pylint: disable=wildcard-import,unused-wildcard-import
from bashi.version.dependencies.clang_cuda import ClangCudaSDKSupport
from alpaka_bashi.alpaka_filter import (
    AlpakaFilter,
    check_only_valid_backend_combinations_a1,
    check_cuda_sdk_host_compiler_a2,
    check_debug_build_hip_a3,
    check_ubuntu_22_04_specifics_a4,
    check_clang_16_and_older_a5,
    check_existing_clang_cuda_for_cuda_sdk_version_a6,
)
import alpaka_bashi.runtime_info
from alpaka_bashi.globals import (
    RT_HOST_COMPILER_CUDA_SUPPORT,
    RT_CLANG_CUDA_MAX_CUDA_SUPPORT,
    BUILD_TYPE,
    CMAKE_RELEASE,
    CMAKE_DEBUG,
)
from alpaka_bashi.versions import get_alpaka_version_relation
from utils import parse_bashi_row


class TestAlpakaFilter(unittest.TestCase):
    VALID_BACKEND_COMBINATIONS = [
        [(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)],
        [(DEVICE_COMPILER, GCC, 8), (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON)],
        [
            (DEVICE_COMPILER, GCC, 8),
            (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
            (HOST_COMPILER, GCC, 8),
        ],
        [(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4)],
        [
            (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
            (HOST_COMPILER, GCC, 11),
            (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON),
            (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
            (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
            (DEVICE_COMPILER, GCC, 11),
            (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON),
        ],
    ]

    def test_valid_backend_combinations_a1(self):
        for row in self.VALID_BACKEND_COMBINATIONS:
            with self.subTest(row=row):
                self.assertTrue(
                    check_only_valid_backend_combinations_a1(parse_bashi_row(row), AlpakaFilter()),
                    f"{row}",
                )
                self.assertTrue(AlpakaFilter()(parse_bashi_row(row)), f"{row}")

    INVALID_BACKEND_COMBINATIONS = [
        [(ALPAKA_ACC_GPU_HIP_ENABLE, ON), (ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON)],
        [(DEVICE_COMPILER, CLANG, 12), (ALPAKA_ACC_GPU_HIP_ENABLE, ON)],
        [(ALPAKA_ACC_ONEAPI_GPU_ENABLE, ON), (HOST_COMPILER, HIPCC, 6.3)],
        [
            (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLE, ON),
            (HOST_COMPILER, GCC, 11),
            (ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLE, ON),
            (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, ON),
            (ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLE, ON),
            (DEVICE_COMPILER, GCC, 11),
            (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON),
            (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
        ],
    ]

    def test_invalid_backend_combinations_a1(self):
        for row in self.INVALID_BACKEND_COMBINATIONS:
            with self.subTest(row=row):
                EXPECTED_ERROR_MSG = "No valid backend combination available."

                reason_msg = io.StringIO()
                self.assertFalse(
                    check_only_valid_backend_combinations_a1(
                        parse_bashi_row(row), AlpakaFilter(output=reason_msg)
                    ),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg = io.StringIO()
                self.assertFalse(
                    AlpakaFilter(output=reason_msg)(parse_bashi_row(row)),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

    HOST_COMPILER_NVCC_VERSIONS = {
        GCC: [10, 11, 12, 13],
        CLANG: [15, 16, 17, 18],
        NVCC: ["12.0", "12.1", "12.2", "12.3", "12.4", "12.5", "12.6"],
    }

    INVALID_CUDA_BACKEND_COMBINATIONS_FOR_RT_FILTER = [
        [(HOST_COMPILER, CLANG, 19), (ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4)],
        [(ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4), (HOST_COMPILER, GCC, 14)],
        [(ALPAKA_ACC_GPU_CUDA_ENABLE, 12.7), (HOST_COMPILER, GCC, 99)],
    ]

    def test_invalid_only_cuda_backend_a2(self):
        runtime_info = {
            RT_HOST_COMPILER_CUDA_SUPPORT: alpaka_bashi.runtime_info.get_rt_func_host_compiler_supports_cuda(
                input_versions=self.HOST_COMPILER_NVCC_VERSIONS,
                version_relation=get_alpaka_version_relation(),
            )
        }

        rt_host_compiler_cuda_support = cast(
            alpaka_bashi.runtime_info.HostCompilerSupportsCuda,
            runtime_info[RT_HOST_COMPILER_CUDA_SUPPORT],
        )

        self.assertEqual(
            rt_host_compiler_cuda_support.get_max_version(GCC), packaging.version.parse("13")
        )
        self.assertEqual(
            rt_host_compiler_cuda_support.get_max_version(CLANG), packaging.version.parse("18")
        )

        for untyped_row in self.INVALID_CUDA_BACKEND_COMBINATIONS_FOR_RT_FILTER:
            row = parse_bashi_row(untyped_row)
            with self.subTest(row=row):
                EXPECTED_ERROR_MSG = (
                    "Only backend combinations with CUDA backend possible. There is no CUDA SDK "
                    f"version, which supports the host compiler {row[HOST_COMPILER].name}-"
                    f"{row[HOST_COMPILER].version}"
                )

                reason_msg = io.StringIO()
                self.assertFalse(
                    check_cuda_sdk_host_compiler_a2(
                        row,
                        AlpakaFilter(output=reason_msg, runtime_infos=runtime_info),
                    ),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg = io.StringIO()
                self.assertFalse(
                    AlpakaFilter(output=reason_msg, runtime_infos=runtime_info)(row),
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

    VALID_HIPCC_BUILD_CONFIGURATIONS = [
        [(DEVICE_COMPILER, HIPCC, 6.2), (BUILD_TYPE, CMAKE_RELEASE)],
        [(HOST_COMPILER, HIPCC, 6.2), (BUILD_TYPE, CMAKE_RELEASE)],
        [(HOST_COMPILER, HIPCC, 6.1), (BUILD_TYPE, CMAKE_DEBUG)],
        [(HOST_COMPILER, HIPCC, 6.3), (BUILD_TYPE, CMAKE_DEBUG)],
        [(HOST_COMPILER, HIPCC, 6.3), (BUILD_TYPE, CMAKE_RELEASE)],
    ]

    def test_valid_hipcc62_debug_build_a3(self):
        for row in self.VALID_HIPCC_BUILD_CONFIGURATIONS:
            with self.subTest(row=row):
                self.assertTrue(
                    check_debug_build_hip_a3(parse_bashi_row(row), AlpakaFilter()),
                    f"{row}",
                )
                self.assertTrue(AlpakaFilter()(parse_bashi_row(row)), f"{row}")

    INVALID_HIPCC_BUILD_CONFIGURATIONS = [
        [(DEVICE_COMPILER, HIPCC, 6.2), (BUILD_TYPE, CMAKE_DEBUG)],
        [(BUILD_TYPE, CMAKE_DEBUG), (HOST_COMPILER, HIPCC, 6.2)],
    ]

    def test_invalid_hipcc62_debug_build_a3(self):
        for row in self.INVALID_HIPCC_BUILD_CONFIGURATIONS:
            with self.subTest(row=row):
                EXPECTED_ERROR_MSG = "Debug builds with HIP/ROCm 6.2 produce compiler errors."

                reason_msg = io.StringIO()
                self.assertFalse(
                    check_debug_build_hip_a3(parse_bashi_row(row), AlpakaFilter(output=reason_msg)),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg = io.StringIO()
                self.assertFalse(
                    AlpakaFilter(output=reason_msg)(parse_bashi_row(row)),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

    VALID_AVAILABLE_COMPILERS_ON_UBUNTU = [
        [(DEVICE_COMPILER, GCC, 14), (UBUNTU, "24.04")],
        [(DEVICE_COMPILER, CLANG, 17), (UBUNTU, "24.04")],
        [
            (HOST_COMPILER, ICPX, "2025.0.1"),
            (DEVICE_COMPILER, ICPX, "2025.0.1"),
            (UBUNTU, "24.04"),
        ],
        [(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON), (UBUNTU, "24.04")],
        [(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, ON), (UBUNTU, "22.04")],
        [(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, OFF), (UBUNTU, "22.04")],
        [(ALPAKA_ACC_GPU_CUDA_ENABLE, 12.4), (UBUNTU, "24.04")],
        [
            (HOST_COMPILER, HIPCC, 6.0),
            (DEVICE_COMPILER, HIPCC, 6.0),
            (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
            (UBUNTU, "22.04"),
        ],
        [
            (HOST_COMPILER, HIPCC, 6.3),
            (DEVICE_COMPILER, HIPCC, 6.3),
            (ALPAKA_ACC_GPU_HIP_ENABLE, ON),
            (UBUNTU, "24.04"),
        ],
    ]

    def test_valid_ubuntu2204_a4(self):
        for row in self.VALID_AVAILABLE_COMPILERS_ON_UBUNTU:
            with self.subTest(row=row):
                self.assertTrue(
                    check_ubuntu_22_04_specifics_a4(parse_bashi_row(row), AlpakaFilter()),
                    f"{row}",
                )
                self.assertTrue(AlpakaFilter()(parse_bashi_row(row)), f"{row}")

    INVALID_AVAILABLE_COMPILERS_ON_UBUNTU = [
        [(DEVICE_COMPILER, GCC, 14), (UBUNTU, "22.04")],
        [(HOST_COMPILER, CLANG, 14), (DEVICE_COMPILER, NVCC, 12.4), (UBUNTU, "22.04")],
        [
            (HOST_COMPILER, ICPX, "2025.2.1"),
            (UBUNTU, "22.04"),
            (DEVICE_COMPILER, ICPX, "2025.2.1"),
        ],
    ]

    def test_invalid_ubuntu2204_generic_compiler_a4(self):
        for row in self.INVALID_AVAILABLE_COMPILERS_ON_UBUNTU:
            with self.subTest(row=row):
                EXPECTED_ERROR_MSG = "Only HIPCC and Clang will be tested on Ubuntu 22.04"

                reason_msg = io.StringIO()
                self.assertFalse(
                    check_ubuntu_22_04_specifics_a4(
                        parse_bashi_row(row), AlpakaFilter(output=reason_msg)
                    ),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg = io.StringIO()
                self.assertFalse(
                    AlpakaFilter(output=reason_msg)(parse_bashi_row(row)),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

    CLANG_COMPILER_VERSIONS_WHICH_ARE_NOT_AVAILABLE_ON_UBUNTU_2204 = [
        [(DEVICE_COMPILER, CLANG, 17), (UBUNTU, "22.04")],
        [(HOST_COMPILER, CLANG, 19), (UBUNTU, "22.04")],
    ]

    def test_invalid_ubuntu2204_clang_compiler_a4(self):
        for row in self.CLANG_COMPILER_VERSIONS_WHICH_ARE_NOT_AVAILABLE_ON_UBUNTU_2204:
            with self.subTest(row=row):
                EXPECTED_ERROR_MSG = "Clang 17 and later will be tested on Ubuntu 24.04 and later."

                reason_msg = io.StringIO()
                self.assertFalse(
                    check_ubuntu_22_04_specifics_a4(
                        parse_bashi_row(row), AlpakaFilter(output=reason_msg)
                    ),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg = io.StringIO()
                self.assertFalse(
                    AlpakaFilter(output=reason_msg)(parse_bashi_row(row)),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

    BACKENDS_ARE_NOT_AVAILABLE_ON_UBUNTU_2204 = [
        ALPAKA_ACC_ONEAPI_CPU_ENABLE,
        ALPAKA_ACC_ONEAPI_GPU_ENABLE,
        ALPAKA_ACC_ONEAPI_FPGA_ENABLE,
        ALPAKA_ACC_GPU_CUDA_ENABLE,
    ]

    def test_invalid_ubuntu2204_backends_a4(self):
        for backend in self.BACKENDS_ARE_NOT_AVAILABLE_ON_UBUNTU_2204:
            with self.subTest(backend=backend):
                row = [(backend, ON), (UBUNTU, "22.04")]
                EXPECTED_ERROR_MSG = (
                    f"The backend {backend} will be not used on Ubuntu 22.04 and older."
                )

                reason_msg = io.StringIO()
                self.assertFalse(
                    check_ubuntu_22_04_specifics_a4(
                        parse_bashi_row(row), AlpakaFilter(output=reason_msg)
                    ),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg = io.StringIO()
                self.assertFalse(
                    AlpakaFilter(output=reason_msg)(parse_bashi_row(row)),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

    AVAILABLE_CLANG_VERSIONS_ON_UBUNTU_2204 = [
        [(DEVICE_COMPILER, CLANG, 14), (UBUNTU, "22.04")],
        [(DEVICE_COMPILER, CLANG, 16), (UBUNTU, "22.04")],
        [(HOST_COMPILER, CLANG, 15), (UBUNTU, "22.04")],
    ]

    def test_valid_clang_ubuntu2204_a5(self):
        for row in self.AVAILABLE_CLANG_VERSIONS_ON_UBUNTU_2204:
            with self.subTest(row=row):
                self.assertTrue(
                    check_clang_16_and_older_a5(parse_bashi_row(row), AlpakaFilter()), f"{row}"
                )
                self.assertTrue(AlpakaFilter()(parse_bashi_row(row)), f"{row}")

    def test_invalid_clang_ubuntu2204_wrong_ubuntu_a5(self):
        for clang_version, ubuntu_version in itertools.product((14, 16), ("24.4", "26.4")):
            for row in [
                [(HOST_COMPILER, CLANG, clang_version), (UBUNTU, ubuntu_version)],
                [(DEVICE_COMPILER, CLANG, clang_version), (UBUNTU, ubuntu_version)],
            ]:
                with self.subTest(row=row):
                    EXPECTED_ERROR_MSG = (
                        f"Clang {clang_version} does not support libc++-13 and later "
                        f"of the host compiler of Ubuntu {ubuntu_version}"
                    )

                    reason_msg = io.StringIO()
                    self.assertFalse(
                        check_clang_16_and_older_a5(
                            parse_bashi_row(row), AlpakaFilter(output=reason_msg)
                        ),
                        f"{row}",
                    )
                    self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                    reason_msg = io.StringIO()
                    self.assertFalse(
                        AlpakaFilter(output=reason_msg)(parse_bashi_row(row)),
                        f"{row}",
                    )
                    self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

    NOT_AVAILABLE_NVCC_CLANG_VERSIONS_ON_UBUNTU_2204 = [
        [(HOST_COMPILER, CLANG, 14), (DEVICE_COMPILER, NVCC, "12.0")],
        [(HOST_COMPILER, CLANG, 16), (DEVICE_COMPILER, NVCC, "12.6")],
    ]

    def test_invalid_clang_ubuntu2204_nvcc12_a5(self):
        for row in self.NOT_AVAILABLE_NVCC_CLANG_VERSIONS_ON_UBUNTU_2204:
            with self.subTest(row=row):
                EXPECTED_ERROR_MSG = (
                    f"NVCC {row[1][2]} is only available on UBUNTU 24.04 "
                    f"and later but Clang {row[0][2]} does not support 24.04 "
                    "and later."
                )

                reason_msg = io.StringIO()
                self.assertFalse(
                    check_clang_16_and_older_a5(
                        parse_bashi_row(row), AlpakaFilter(output=reason_msg)
                    ),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg = io.StringIO()
                self.assertFalse(
                    AlpakaFilter(output=reason_msg)(parse_bashi_row(row)),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

    NOT_AVAILABLE_CUDA_SDK_CLANG_VERSIONS_ON_UBUNTU_2204 = [
        [(HOST_COMPILER, CLANG, 14), (ALPAKA_ACC_GPU_CUDA_ENABLE, "12.0")],
        [(HOST_COMPILER, CLANG, 16), (ALPAKA_ACC_GPU_CUDA_ENABLE, "12.6")],
    ]

    def test_invalid_clang_ubuntu2204_cuda12_a5(self):
        for row in self.NOT_AVAILABLE_CUDA_SDK_CLANG_VERSIONS_ON_UBUNTU_2204:
            with self.subTest(row=row):
                EXPECTED_ERROR_MSG = (
                    f"CUDA {row[1][1]} is only available on UBUNTU 24.04 "
                    f"and later but Clang {row[0][2]} does not support 24.04 "
                    "and later."
                )

                reason_msg = io.StringIO()
                self.assertFalse(
                    check_clang_16_and_older_a5(
                        parse_bashi_row(row), AlpakaFilter(output=reason_msg)
                    ),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg = io.StringIO()
                self.assertFalse(
                    AlpakaFilter(output=reason_msg)(parse_bashi_row(row)),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

    def test_invalid_clang_ubuntu2204_disabled_cpu_backend_a5(self):
        row = (HOST_COMPILER, CLANG, 14), (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLE, OFF)
        EXPECTED_ERROR_MSG = f"Clang {row[0][2]} works only together with CPU backends."

        reason_msg = io.StringIO()
        self.assertFalse(
            check_clang_16_and_older_a5(parse_bashi_row(row), AlpakaFilter(output=reason_msg)),
            f"{row}",
        )
        self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

        reason_msg = io.StringIO()
        self.assertFalse(
            AlpakaFilter(output=reason_msg)(parse_bashi_row(row)),
            f"{row}",
        )
        self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

    CLANG_CUDA_VERSIONS = {CLANG_CUDA: [14, 15, 16, 17, 18, 19]}

    TEST_CLANG_CUDA_SDK_SUPPORT_TABLE = [
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

    VALID_CLANG_CUDA_SDK_COMBINATIONS = [
        [(ALPAKA_ACC_GPU_CUDA_ENABLE, 12.0), (HOST_COMPILER, CLANG_CUDA, 17)],
        [(ALPAKA_ACC_GPU_CUDA_ENABLE, 11.5), (DEVICE_COMPILER, CLANG_CUDA, 14)],
    ]

    def test_valid_clang_cuda_cuda_sdk_a6(self):
        runtime_info: Dict[str, Callable[..., bool]] = {}
        runtime_info[RT_CLANG_CUDA_MAX_CUDA_SUPPORT] = (
            alpaka_bashi.runtime_info.ClangCUDAMaxSupportsCuda(
                bashi.VersionRelation(
                    clang_cuda_max_cuda_version=self.TEST_CLANG_CUDA_SDK_SUPPORT_TABLE
                ),
                self.CLANG_CUDA_VERSIONS,
            )
        )

        for row in self.VALID_CLANG_CUDA_SDK_COMBINATIONS:
            with self.subTest(row=row):
                self.assertTrue(
                    check_existing_clang_cuda_for_cuda_sdk_version_a6(
                        parse_bashi_row(row), AlpakaFilter(runtime_infos=runtime_info)
                    ),
                    f"{row}",
                )
                self.assertTrue(
                    AlpakaFilter(runtime_infos=runtime_info)(parse_bashi_row(row)),
                    f"{row}",
                )

    INVALID_CLANG_CUDA_SDK_COMBINATIONS = [
        [(ALPAKA_ACC_GPU_CUDA_ENABLE, 12.8), (HOST_COMPILER, CLANG_CUDA, 17)],
        [(ALPAKA_ACC_GPU_CUDA_ENABLE, 13.0), (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLE, OFF_VER)],
        [(ALPAKA_ACC_GPU_CUDA_ENABLE, 13.2), (DEVICE_COMPILER, CLANG_CUDA, 19)],
    ]

    def test_invalid_clang_cuda_cuda_sdk_a6(self):
        runtime_info: Dict[str, Callable[..., bool]] = {}
        runtime_info[RT_CLANG_CUDA_MAX_CUDA_SUPPORT] = (
            alpaka_bashi.runtime_info.ClangCUDAMaxSupportsCuda(
                bashi.VersionRelation(
                    clang_cuda_max_cuda_version=self.TEST_CLANG_CUDA_SDK_SUPPORT_TABLE
                ),
                self.CLANG_CUDA_VERSIONS,
            )
        )

        for row in self.INVALID_CLANG_CUDA_SDK_COMBINATIONS:
            with self.subTest(row=row):
                EXPECTED_ERROR_MSG = (
                    "There is no Clang-CUDA version in the combination list, which supports the "
                    f"CUDA {row[0][1]} SDK."
                )

                reason_msg = io.StringIO()
                self.assertFalse(
                    check_existing_clang_cuda_for_cuda_sdk_version_a6(
                        parse_bashi_row(row),
                        AlpakaFilter(output=reason_msg, runtime_infos=runtime_info),
                    ),
                    f"{row}",
                )
                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")

                reason_msg = io.StringIO()
                self.assertFalse(
                    AlpakaFilter(output=reason_msg, runtime_infos=runtime_info)(
                        parse_bashi_row(row)
                    ),
                    f"{row}",
                )

                self.assertEqual(reason_msg.getvalue(), EXPECTED_ERROR_MSG, f"{row}")
