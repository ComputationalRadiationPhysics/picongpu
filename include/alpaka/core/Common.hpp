/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/core/Debug.hpp>

#include <boost/predef/version_number.h>

// Boost.Uuid errors with VS2017 when intrin.h is not included
#if defined(_MSC_VER) && _MSC_VER >= 1910
    #include <intrin.h>
#endif

//#############################################################################
// This extends Boost.Predef by detecting:
// - BOOST_LANG_CUDA
// - BOOST_ARCH_CUDA_DEVICE
// - BOOST_COMP_NVCC
// - BOOST_COMP_CLANG_CUDA

//-----------------------------------------------------------------------------
// BOOST_PREDEF_MAKE_10_VVRRP(V)
#define BOOST_PREDEF_MAKE_10_VVRRP(V) BOOST_VERSION_NUMBER(((V)/1000)%100,((V)/10)%100,(V)%10)

//-----------------------------------------------------------------------------
// CUDA language detection
// - clang defines __CUDA__ and __CUDACC__ when compiling CUDA code ('-x cuda')
// - nvcc defines __CUDACC__ when compiling CUDA code
#if defined(__CUDA__) || defined(__CUDACC__)
    #include <cuda.h>
    #define BOOST_LANG_CUDA BOOST_PREDEF_MAKE_10_VVRRP(CUDA_VERSION)
#else
    #define BOOST_LANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#endif

//-----------------------------------------------------------------------------
// CUDA device architecture detection
#if defined(__CUDA_ARCH__)
    #define BOOST_ARCH_CUDA_DEVICE BOOST_PREDEF_MAKE_10_VRP(__CUDA_ARCH__)
#else
    #define BOOST_ARCH_CUDA_DEVICE BOOST_VERSION_NUMBER_NOT_AVAILABLE
#endif

//-----------------------------------------------------------------------------
// nvcc CUDA compiler detection
#if defined(__CUDACC__) && defined(__NVCC__)
    // The __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__ and __CUDACC_VER_BUILD__
    // have been added with nvcc 7.5 and have not been available before.
    #if !defined(__CUDACC_VER_MAJOR__) || !defined(__CUDACC_VER_MINOR__) || !defined(__CUDACC_VER_BUILD__)
        #define BOOST_COMP_NVCC BOOST_VERSION_NUMBER_AVAILABLE
    #else
        #define BOOST_COMP_NVCC BOOST_VERSION_NUMBER(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__)
    #endif
#else
    #define BOOST_COMP_NVCC BOOST_VERSION_NUMBER_NOT_AVAILABLE
#endif

//-----------------------------------------------------------------------------
// clang CUDA compiler detection
// Currently __CUDA__ is only defined by clang when compiling CUDA code.
#if defined(__clang__) && defined(__CUDA__)
    #define BOOST_COMP_CLANG_CUDA BOOST_COMP_CLANG
#else
    #define BOOST_COMP_CLANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
#endif

//-----------------------------------------------------------------------------
// Boost does not yet correctly identify clang when compiling CUDA code.
// After explicitly including <boost/config.hpp> we can safely undefine some of the wrong settings.
#if BOOST_COMP_CLANG_CUDA
    #include <boost/config.hpp>
    #undef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif

//-----------------------------------------------------------------------------
// Boost disables variadic templates for nvcc (in some cases because it was buggy).
// However, we rely on it being enabled, as it was in all previous boost versions we support.
// After explicitly including <boost/config.hpp> we can safely undefine the wrong setting.
#if BOOST_COMP_NVCC
    #include <boost/config.hpp>
    #undef BOOST_NO_CXX11_VARIADIC_TEMPLATES
#endif

//-----------------------------------------------------------------------------
//! All functions that can be used on an accelerator have to be attributed with ALPAKA_FN_ACC_CUDA_ONLY or ALPAKA_FN_ACC.
//!
//! Usage:
//! ALPAKA_FN_ACC
//! auto add(std::int32_t a, std::int32_t b)
//! -> std::int32_t;
#if BOOST_LANG_CUDA
    #define ALPAKA_FN_ACC_CUDA_ONLY __device__
    #define ALPAKA_FN_ACC_NO_CUDA __host__
    #if defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE)
        #define ALPAKA_FN_ACC __device__
    #else
        #define ALPAKA_FN_ACC __device__ __host__
    #endif
    #define ALPAKA_FN_HOST_ACC __device__ __host__
    #define ALPAKA_FN_HOST __host__
#else
    // NOTE: ALPAKA_FN_ACC_CUDA_ONLY should not be defined to cause build failures when CUDA only functions are used and CUDA is disabled.
    // However, this also destroys syntax highlighting.
    #define ALPAKA_FN_ACC_CUDA_ONLY
    #define ALPAKA_FN_ACC_NO_CUDA
    #define ALPAKA_FN_ACC
    #define ALPAKA_FN_HOST_ACC
    #define ALPAKA_FN_HOST
#endif

//-----------------------------------------------------------------------------
//! Disable nvcc warning:
//! 'calling a __host__ function from __host__ __device__ function.'
//!
//! Usage:
//! ALPAKA_NO_HOST_ACC_WARNING
//! ALPAKA_FN_HOST_ACC function_declaration()
//!
//! WARNING: Only use this method if there is no other way.
//! Most cases can be solved by #if BOOST_ARCH_CUDA_DEVICE or #if BOOST_LANG_CUDA.
#if BOOST_LANG_CUDA && !BOOST_COMP_CLANG_CUDA
    #if BOOST_COMP_MSVC
        #define ALPAKA_NO_HOST_ACC_WARNING\
            __pragma(hd_warning_disable)
    #else
        #define ALPAKA_NO_HOST_ACC_WARNING\
            _Pragma("hd_warning_disable")
    #endif
#else
    #define ALPAKA_NO_HOST_ACC_WARNING
#endif

//-----------------------------------------------------------------------------
//! Macro defining the inline function attribute.
#if BOOST_LANG_CUDA
    #define ALPAKA_FN_INLINE __forceinline__
#else
    #define ALPAKA_FN_INLINE inline
#endif

//-----------------------------------------------------------------------------
//! This macro defines a variable lying in global accelerator device memory.
//!
//! Example:
//!   ALPAKA_STATIC_DEV_MEM_GLOBAL int i;
//!
//! Those variables behave like ordinary variables when used in file-scope.
//! They have external linkage (are accessible from other compilation units).
//! If you want to access it from a different compilation unit, you have to declare it as extern:
//!   extern ALPAKA_STATIC_DEV_MEM_GLOBAL int i;
//! Like ordinary variables, only one definition is allowed (ODR)
//! Failure to do so might lead to linker errors.
//!
//! In contrast to ordinary variables, you can not define such variables
//! as static compilation unit local variables with internal linkage
//! because this is forbidden by CUDA.
#if BOOST_LANG_CUDA && BOOST_ARCH_CUDA_DEVICE
    #define ALPAKA_STATIC_DEV_MEM_GLOBAL __device__
#else
    #define ALPAKA_STATIC_DEV_MEM_GLOBAL
#endif

//-----------------------------------------------------------------------------
//! This macro defines a variable lying in constant accelerator device memory.
//!
//! Example:
//!   ALPAKA_STATIC_DEV_MEM_CONSTANT int i;
//!
//! Those variables behave like ordinary variables when used in file-scope.
//! They have external linkage (are accessible from other compilation units).
//! If you want to access it from a different compilation unit, you have to declare it as extern:
//!   extern ALPAKA_STATIC_DEV_MEM_CONSTANT int i;
//! Like ordinary variables, only one definition is allowed (ODR)
//! Failure to do so might lead to linker errors.
//!
//! In contrast to ordinary variables, you can not define such variables
//! as static compilation unit local variables with internal linkage
//! because this is forbidden by CUDA.
#if BOOST_LANG_CUDA && BOOST_ARCH_CUDA_DEVICE
    #define ALPAKA_STATIC_DEV_MEM_CONSTANT __constant__
#else
    #define ALPAKA_STATIC_DEV_MEM_CONSTANT
#endif
