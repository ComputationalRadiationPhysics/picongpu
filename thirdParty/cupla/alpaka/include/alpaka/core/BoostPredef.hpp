/* Copyright 2019 Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <boost/predef.h>

//-----------------------------------------------------------------------------
// In boost since 1.68.0
// BOOST_PREDEF_MAKE_10_VVRRP(V)
#if !defined(BOOST_PREDEF_MAKE_10_VVRRP)
#    define BOOST_PREDEF_MAKE_10_VVRRP(V) BOOST_VERSION_NUMBER(((V) / 1000) % 100, ((V) / 10) % 100, (V) % 10)
#endif

//---------------------------------------HIP-----------------------------------
// __HIPCC__ is defined by hipcc (if either __CUDACC__ is defined)
// https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md#compiler-defines-summary
#if !defined(BOOST_LANG_HIP)
#    if defined(__HIPCC__) && (defined(__CUDACC__) || defined(__HIP__))
#        include <hip/hip_runtime.h>
// HIP defines "abort()" as "{asm("trap;");}", which breaks some kernels
#        undef abort
// there is no HIP_VERSION macro
#        define BOOST_LANG_HIP BOOST_VERSION_NUMBER_AVAILABLE
#        if defined(BOOST_LANG_CUDA) && BOOST_LANG_CUDA
#            undef BOOST_LANG_CUDA
#            define BOOST_LANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#        endif
#    else
#        define BOOST_LANG_HIP BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

//-----------------------------------------------------------------------------
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

//-----------------------------------------------------------------------------
// hip compiler detection
#if !defined(BOOST_COMP_HIP)
#    if defined(__HIP__)
#        define BOOST_COMP_HIP BOOST_VERSION_NUMBER_AVAILABLE
#    else
#        define BOOST_COMP_HIP BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

//-----------------------------------------------------------------------------
// In boost since 1.68.0
// CUDA language detection
// - clang defines __CUDA__ and __CUDACC__ when compiling CUDA code ('-x cuda')
// - nvcc defines __CUDACC__ when compiling CUDA code
#if !defined(BOOST_LANG_CUDA)
#    if defined(__CUDA__) || defined(__CUDACC__)
#        include <cuda.h>
#        define BOOST_LANG_CUDA BOOST_PREDEF_MAKE_10_VVRRP(CUDA_VERSION)
#    else
#        define BOOST_LANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

//-----------------------------------------------------------------------------
// In boost since 1.68.0
// CUDA device architecture detection
#if !defined(BOOST_ARCH_PTX)
#    if defined(__CUDA_ARCH__)
#        define BOOST_ARCH_PTX BOOST_PREDEF_MAKE_10_VRP(__CUDA_ARCH__)
#    else
#        define BOOST_ARCH_PTX BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

//-----------------------------------------------------------------------------
// In boost since 1.68.0
// nvcc CUDA compiler detection

#include <boost/version.hpp>
#if BOOST_VERSION >= 106800
// BOOST_COMP_NVCC_EMULATED is defined by boost instead of BOOST_COMP_NVCC
#    if defined(BOOST_COMP_NVCC) && defined(BOOST_COMP_NVCC_EMULATED)
#        undef BOOST_COMP_NVCC
#        define BOOST_COMP_NVCC BOOST_COMP_NVCC_EMULATED
#    endif
#endif

#if !defined(BOOST_COMP_NVCC)
#    if defined(__NVCC__)
// The __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__ and __CUDACC_VER_BUILD__
// have been added with nvcc 7.5 and have not been available before.
#        if !defined(__CUDACC_VER_MAJOR__) || !defined(__CUDACC_VER_MINOR__) || !defined(__CUDACC_VER_BUILD__)
#            define BOOST_COMP_NVCC BOOST_VERSION_NUMBER_AVAILABLE
#        else
#            define BOOST_COMP_NVCC                                                                                   \
                BOOST_VERSION_NUMBER(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__)
#        endif
#    else
#        define BOOST_COMP_NVCC BOOST_VERSION_NUMBER_NOT_AVAILABLE
#    endif
#endif

//-----------------------------------------------------------------------------
// clang CUDA compiler detection
// Currently __CUDA__ is only defined by clang when compiling CUDA code.
#if defined(__clang__) && defined(__CUDA__)
#    define BOOST_COMP_CLANG_CUDA BOOST_COMP_CLANG
#else
#    define BOOST_COMP_CLANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#endif

//-----------------------------------------------------------------------------
// Intel compiler detection
// BOOST_COMP_INTEL_EMULATED is defined by boost instead of BOOST_COMP_INTEL
#if defined(BOOST_COMP_INTEL) && defined(BOOST_COMP_INTEL_EMULATED)
#    undef BOOST_COMP_INTEL
#    define BOOST_COMP_INTEL BOOST_COMP_INTEL_EMULATED
#endif

//-----------------------------------------------------------------------------
// PGI and NV HPC SDK compiler detection
// BOOST_COMP_PGI_EMULATED is defined by boost instead of BOOST_COMP_PGI
#if defined(BOOST_COMP_PGI) && defined(BOOST_COMP_PGI_EMULATED)
#    undef BOOST_COMP_PGI
#    define BOOST_COMP_PGI BOOST_COMP_PGI_EMULATED
#endif
