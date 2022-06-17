/* Copyright 2022 Benjamin Worpitz, Matthias Werner, Jan Stephan
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <boost/predef.h>

#ifdef __INTEL_COMPILER
#    warning                                                                                                          \
        "The Intel Classic compiler (icpc) is no longer supported. Please upgrade to the Intel LLVM compiler (ipcx)."
#endif

//---------------------------------------HIP-----------------------------------
// __HIPCC__ is defined by hipcc (if either __CUDACC__ is defined)
// https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md#compiler-defines-summary
#if !defined(BOOST_LANG_HIP)
#    if defined(__HIPCC__) && (defined(__CUDACC__) || defined(__HIP__))
#        include <hip/hip_runtime.h>
// HIP defines "abort()" as "{asm("trap;");}", which breaks some kernels
#        undef abort
#        define BOOST_LANG_HIP BOOST_VERSION_NUMBER(HIP_VERSION_MAJOR, HIP_VERSION_MINOR, 0)
#        if defined(BOOST_LANG_CUDA) && BOOST_LANG_CUDA
#            undef BOOST_LANG_CUDA
#            define BOOST_LANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#        endif
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

// hip compiler detection
#if !defined(BOOST_COMP_HIP)
#    if defined(__HIP__)
#        define BOOST_COMP_HIP BOOST_VERSION_NUMBER_AVAILABLE
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
