/* Copyright 2023 Benjamin Worpitz, Matthias Werner, René Widera, Sergei Bastrakov, Jeffrey Kelling,
 *                Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/PP.hpp"

#ifdef __INTEL_COMPILER
#    warning                                                                                                          \
        "The Intel Classic compiler (icpc) is no longer supported. Please upgrade to the Intel LLVM compiler (ipcx)."
#endif

// ######## detect operating systems ########

// WINDOWS
#if !defined(ALPAKA_OS_WINDOWS)
#    if defined(_WIN64) || defined(__MINGW64__)
#        define ALPAKA_OS_WINDOWS 1
#    else
#        define ALPAKA_OS_WINDOWS 0
#    endif
#endif


// Linux
#if !defined(ALPAKA_OS_LINUX)
#    if defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
#        define ALPAKA_OS_LINUX 1
#    else
#        define ALPAKA_OS_LINUX 0
#    endif
#endif

// Apple
#if !defined(ALPAKA_OS_IOS)
#    if defined(__APPLE__)
#        define ALPAKA_OS_IOS 1
#    else
#        define ALPAKA_OS_IOS 0
#    endif
#endif

// Cygwin
#if !defined(ALPAKA_OS_CYGWIN)
#    if defined(__CYGWIN__)
#        define ALPAKA_OS_CYGWIN 1
#    else
#        define ALPAKA_OS_CYGWIN 0
#    endif
#endif

// ### architectures

// X86
#if !defined(ALPAKA_ARCH_X86)
#    if defined(__x86_64__) || defined(_M_X64)
#        define ALPAKA_ARCH_X86 1
#    else
#        define ALPAKA_ARCH_X86 0
#    endif
#endif

// RISCV
#if !defined(ALPAKA_ARCH_RISCV)
#    if defined(__riscv)
#        define ALPAKA_ARCH_RISCV 1
#    else
#        define ALPAKA_ARCH_RISCV 0
#    endif
#endif

// ARM
#if !defined(ALPAKA_ARCH_ARM)
#    if defined(__ARM_ARCH) || defined(__arm__) || defined(__arm64)
#        define ALPAKA_ARCH_ARM 1
#    else
#        define ALPAKA_ARCH_ARM 0
#    endif
#endif

// NVIDIA device compile
// ALPAKA_ARCH_PTX is a misnomer as it holds the compute capability version (the virtual GPU architecture) and not the
// PTX ISA version. This misnomer exists in CUDA as well. https://github.com/NVIDIA/cccl/issues/4081
#if !defined(ALPAKA_ARCH_PTX)
#    if defined(__CUDA_ARCH__)
#        define ALPAKA_ARCH_PTX ALPAKA_VRP_TO_VERSION(__CUDA_ARCH__)
#    else
#        define ALPAKA_ARCH_PTX ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

/** HIP device compile
 *
 * The version on the host side will always be ALPAKA_VERSION_NUMBER_NOT_AVAILABLE.
 * On the device side unknown version will be set to ALPAKA_VERSION_NUMBER_NOT_AVAILABLE.
 *
 *  Rules:
 *   - the last two digits will be handled as HEX values and support 0-9 and a-f
 *   - gfx9xy (numeric): 9xy -> ALPAKA_VERSION_NUMBER(9,x,y)
 *   - gfx10xy / gfx11xy: stxy -> ALPAKA_VERSION_NUMBER(st,x,y)
 *   - Suffix: a == 10, b == 11, c == 12
 *      - gfx90a -> ALPAKA_VERSION_NUMBER(9,0,10)
 *      - gfx90c -> ALPAKA_VERSION_NUMBER(9,0,12)
 */
#if !defined(ALPAKA_ARCH_AMD)
#    if defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1
#        define ALPAKA_ARCH_AMD ALPAKA_AMDGPU_ARCH
#    else
#        define ALPAKA_ARCH_AMD ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

/** HSA device compile
 *
 * This version is equal to ALPAKA_ARCH_AMD and is only carried for backwards compatibility to older ALPAKA versions.
 */
#define ALPAKA_ARCH_HSA ALPAKA_ARCH_AMD

// ######## compiler ########

// HIP compiler detection
#if !defined(ALPAKA_COMP_HIP)
#    if defined(__HIP__) // Defined by hip-clang and vanilla clang in HIP mode.
#        include <hip/hip_version.h>
// HIP doesn't give us a patch level for the last entry, just a gitdate
#        define ALPAKA_COMP_HIP ALPAKA_VERSION_NUMBER(HIP_VERSION_MAJOR, HIP_VERSION_MINOR, 0)
#    else
#        define ALPAKA_COMP_HIP ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// nvcc compiler
#if defined(__NVCC__)
#    define ALPAKA_COMP_NVCC ALPAKA_VERSION_NUMBER(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__)
#else
#    define ALPAKA_COMP_NVCC ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#endif

// clang compiler
#if defined(__clang__)
#    define ALPAKA_COMP_CLANG ALPAKA_VERSION_NUMBER(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
#    define ALPAKA_COMP_CLANG ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#endif

// MSVC compiler
#if defined(_MSC_VER)
#    define ALPAKA_COMP_MSVC                                                                                          \
        ALPAKA_VERSION_NUMBER((_MSC_FULL_VER) % 10'000'000, ((_MSC_FULL_VER) / 100000) % 100, (_MSC_FULL_VER) % 100000)
#else
#    define ALPAKA_COMP_MSVC ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#endif

// gnu compiler (excluding compilers which emulates gnu compiler like clang)
#if defined(__GNUC__) && !defined(__clang__)
#    if defined(__GNUC_PATCHLEVEL__)
#        define ALPAKA_COMP_GNUC ALPAKA_VERSION_NUMBER(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__)
#    else
#        define ALPAKA_COMP_GNUC ALPAKA_VERSION_NUMBER(__GNUC__, __GNUC_MINOR__, 0)
#    endif
#else
#    define ALPAKA_COMP_GNUC ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#endif

// IBM compiler
// only clang based is supported
#if defined(__ibmxl__)
#    define ALPAKA_COMP_IBM ALPAKA_VERSION_NUMBER(__ibmxl_version__, __ibmxl_release__, __ibmxl_modification__)
#else
#    define ALPAKA_COMP_IBM ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#endif

// clang CUDA compiler detection
// Currently __CUDA__ is only defined by clang when compiling CUDA code.
#if defined(__clang__) && defined(__CUDA__)
#    define ALPAKA_COMP_CLANG_CUDA ALPAKA_VERSION_NUMBER(__clang_major__, __clang_minor__, __clang_patchlevel__)
#else
#    define ALPAKA_COMP_CLANG_CUDA ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#endif

// PGI and NV HPC SDK compiler detection
#if defined(__PGI)
#    define ALPAKA_COMP_PGI ALPAKA_VERSION_NUMBER(__PGIC__, __PGIC_MINOR__, __PGIC_PATCHLEVEL__)
#else
#    define ALPAKA_COMP_PGI ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#endif

// Intel LLVM compiler detection
#if !defined(ALPAKA_COMP_ICPX)
#    if defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
// The version string for icpx 2023.1.0 is 20230100. In Boost.Predef this becomes (53,1,0).
#        define ALPAKA_COMP_ICPX ALPAKA_YYYYMMDD_TO_VERSION(__INTEL_LLVM_COMPILER)
#    else
#        define ALPAKA_COMP_ICPX ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// ######## C++ language ########

//---------------------------------------HIP-----------------------------------
// __HIP__ is defined by both hip-clang and vanilla clang in HIP mode.
// https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md#compiler-defines-summary
#if !defined(ALPAKA_LANG_HIP)
#    if defined(__HIP__)
#        include <hip/hip_version.h>
// HIP doesn't give us a patch level for the last entry, just a gitdate
#        define ALPAKA_LANG_HIP ALPAKA_VERSION_NUMBER(HIP_VERSION_MAJOR, HIP_VERSION_MINOR, 0)
#    else
#        define ALPAKA_LANG_HIP ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// CUDA
#if !defined(ALPAKA_LANG_CUDA)
#    if defined(__CUDACC__) || defined(__CUDA__)
#        include <cuda.h>
// CUDA doesn't give us a patch level for the last entry, just zero.
#        define ALPAKA_LANG_CUDA ALPAKA_VVRRP_TO_VERSION(CUDART_VERSION)
#    else
#        define ALPAKA_LANG_CUDA ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// Intel OneAPI Sycl GPU
#if !defined(ALPAKA_LANG_SYCL)
#    if defined(SYCL_LANGUAGE_VERSION)
#        define ALPAKA_LANG_SYCL ALPAKA_YYYYMMDD_TO_VERSION(SYCL_LANGUAGE_VERSION)
#    else
#        define ALPAKA_LANG_SYCL ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#    if(ALPAKA_COMP_ICPX)
// ONE API must be detected via the ICPX compiler see
// https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2023-2/use-predefined-macros-to-specify-intel-compilers.html
#        define ALPAKA_LANG_ONEAPI ALPAKA_COMP_ICPX
#    endif
#endif

// OpenMP
#if !defined(ALPAKA_OMP)
#    if defined(_OPENMP)
#        include <omp.h>
#    endif
#    if defined(_OPENMP)
#        define ALPAKA_OMP ALPAKA_YYYYMM_TO_VERSION(_OPENMP)
#    else
#        define ALPAKA_OMP ALPAKA_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif
