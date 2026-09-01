/* Copyright 2022  Andrea Bocci, Mehmet Yusufoglu, René Widera, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Implementation details
#include "alpaka/core/Sycl.hpp"

#include <stdexcept>

//! ALPAKA_THROW_ACC either aborts(terminating the program and creating a core dump) or throws std::runtime_error
//! depending on the Acc. The std::runtime_error exception can be caught in the catch block.
//!
//! For CUDA __trap function is used which triggers std::runtime_error but can be caught during wait not exec.
//! For HIP abort() function is used and calls __builtin_trap()
//! For SYCL assert(false) is not used since it can be disabled with the -DNDEBUG compile option. abort() is used
//! although it generates a runtime error instead of aborting in GPUs:
//! "Caught synchronous SYCL exception: Unresolved Symbol <abort> -999 (Unknown PI error)."
//!
//! The OpenMP specification mandates that exceptions thrown by some thread must be handled by the same thread.
//! Therefore std::runtime_error thrown by ALPAKA_THROW_ACC aborts the the program for OpenMP backends. If needed
//! the SIGABRT signal can be caught by signal handler.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && defined(__CUDA_ARCH__)
#    define ALPAKA_THROW_ACC(MSG)                                                                                     \
        {                                                                                                             \
            printf(                                                                                                   \
                "alpaka encountered a user-defined error condition while running on the CUDA back-end:\n%s",          \
                (MSG));                                                                                               \
            __trap();                                                                                                 \
        }
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED) && defined(__HIP_DEVICE_COMPILE__)
#    define ALPAKA_THROW_ACC(MSG)                                                                                     \
        {                                                                                                             \
            printf(                                                                                                   \
                "alpaka encountered a user-defined error condition while running on the HIP back-end:\n%s",           \
                (MSG));                                                                                               \
            abort();                                                                                                  \
        }
#elif defined(ALPAKA_ACC_SYCL_ENABLED) && defined(__SYCL_DEVICE_ONLY__)
#    define ALPAKA_THROW_ACC(MSG)                                                                                     \
        {                                                                                                             \
            printf(                                                                                                   \
                "alpaka encountered a user-defined error condition while running on the SYCL back-end:\n%s",          \
                (MSG));                                                                                               \
            abort();                                                                                                  \
        }
#elif defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE) || defined(ALPAKA_ACC_GPU_HIP_ONLY_MODE)
// In CUDA-only or HIP-only mode, ALPAKA_FN_ACC functions are __device__ functions, and do not support having a throw
// visible even during host compilation
#    define ALPAKA_THROW_ACC(MSG)                                                                                     \
        {                                                                                                             \
        }
#else
#    define ALPAKA_THROW_ACC(MSG)                                                                                     \
        {                                                                                                             \
            printf("alpaka encountered a user-defined error condition:\n%s", (MSG));                                  \
            throw std::runtime_error(MSG);                                                                            \
        }
#endif
