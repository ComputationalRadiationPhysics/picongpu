/* Copyright 2023 Benjamin Worpitz, Matthias Werner, René Widera, Sergei Bastrakov, Jeffrey Kelling,
 *                Bernhard Manfred Gruber, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <boost/predef.h>

#ifdef __INTEL_COMPILER
#    warning                                                                                                          \
        "The Intel Classic compiler (icpc) is no longer supported. Please upgrade to the Intel LLVM compiler (ipcx)."
#endif

//---------------------------------------HIP-----------------------------------
// __HIP__ is defined by both hip-clang and vanilla clang in HIP mode.
// https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md#compiler-defines-summary
#if !defined(BOOST_LANG_HIP)
#    if defined(__HIP__)
/* BOOST_LANG_CUDA is enabled when either __CUDACC__ (nvcc) or __CUDA__ (clang) are defined. This occurs when
   nvcc / clang encounter a CUDA source file. Since there are no HIP source files we treat every source file
   as HIP when we are using a HIP-capable compiler. */
#        include <hip/hip_version.h>
// HIP doesn't give us a patch level for the last entry, just a gitdate
#        define BOOST_LANG_HIP BOOST_VERSION_NUMBER(HIP_VERSION_MAJOR, HIP_VERSION_MINOR, 0)
#    else
#        define BOOST_LANG_HIP BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// HSA device architecture detection (HSA generated via HIP(clang))
#if !defined(BOOST_ARCH_HSA)
#    if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__ == 1 && defined(__HIP__)
// __HIP_DEVICE_COMPILE__ does not represent feature capability of target device like CUDA_ARCH.
// For feature detection there are special macros, see ROCm's HIP porting guide.
#        define BOOST_ARCH_HSA BOOST_VERSION_NUMBER_AVAILABLE
#    else
#        define BOOST_ARCH_HSA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// HIP compiler detection
#if !defined(BOOST_COMP_HIP)
#    if defined(__HIP__) // Defined by hip-clang and vanilla clang in HIP mode.
#        include <hip/hip_version.h>
// HIP doesn't give us a patch level for the last entry, just a gitdate
#        define BOOST_COMP_HIP BOOST_VERSION_NUMBER(HIP_VERSION_MAJOR, HIP_VERSION_MINOR, 0)
#    else
#        define BOOST_COMP_HIP BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

// clang CUDA compiler detection
// Currently __CUDA__ is only defined by clang when compiling CUDA code.
#if defined(__clang__) && defined(__CUDA__)
#    define BOOST_COMP_CLANG_CUDA BOOST_COMP_CLANG
#else
#    define BOOST_COMP_CLANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#endif

// PGI and NV HPC SDK compiler detection
// As of Boost 1.74, Boost.Predef's compiler detection is a bit weird. Recent PGI compilers will be identified as
// BOOST_COMP_PGI_EMULATED. Boost.Predef has lackluster front-end support and mistakes the EDG front-end
// for an actual compiler.
// TODO: Whenever you look at this code please check whether https://github.com/boostorg/predef/issues/28 and
// https://github.com/boostorg/predef/issues/51 have been resolved.
// BOOST_COMP_PGI_EMULATED is defined by boost instead of BOOST_COMP_PGI
#if defined(BOOST_COMP_PGI) && defined(BOOST_COMP_PGI_EMULATED)
#    undef BOOST_COMP_PGI
#    define BOOST_COMP_PGI BOOST_COMP_PGI_EMULATED
#endif

// Intel LLVM compiler detection
#if !defined(BOOST_COMP_ICPX)
#    if defined(SYCL_LANGUAGE_VERSION) && defined(__INTEL_LLVM_COMPILER)
// The version string for icpx 2023.1.0 is 20230100. In Boost.Predef this becomes (53,1,0).
#        define BOOST_COMP_ICPX BOOST_PREDEF_MAKE_YYYYMMDD(__INTEL_LLVM_COMPILER)
#    endif
#endif
