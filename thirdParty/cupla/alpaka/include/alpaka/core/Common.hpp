/* Copyright 2019 Axel Huebl, Benjamin Worpitz, Matthias Werner
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/BoostPredef.hpp>
#include <alpaka/core/Debug.hpp>

// Boost.Uuid errors with VS2017 when intrin.h is not included
#if defined(_MSC_VER) && _MSC_VER >= 1910
#    include <intrin.h>
#endif

//-----------------------------------------------------------------------------
//! All functions that can be used on an accelerator have to be attributed with ALPAKA_FN_ACC or ALPAKA_FN_HOST_ACC.
//!
//! Usage:
//! ALPAKA_FN_ACC
//! auto add(std::int32_t a, std::int32_t b)
//! -> std::int32_t;
#if BOOST_LANG_CUDA || BOOST_LANG_HIP
#    if defined(ALPAKA_ACC_GPU_CUDA_ONLY_MODE) || defined(ALPAKA_ACC_GPU_HIP_ONLY_MODE)
#        define ALPAKA_FN_ACC __device__
#    else
#        define ALPAKA_FN_ACC __device__ __host__
#    endif
#    define ALPAKA_FN_HOST_ACC __device__ __host__
#    define ALPAKA_FN_HOST __host__
#else
#    define ALPAKA_FN_ACC
#    define ALPAKA_FN_HOST_ACC
#    define ALPAKA_FN_HOST
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
//! Most cases can be solved by #if BOOST_ARCH_PTX or #if BOOST_LANG_CUDA.
#if(BOOST_LANG_CUDA && !BOOST_COMP_CLANG_CUDA) || BOOST_LANG_HIP
#    if BOOST_COMP_MSVC || defined(BOOST_COMP_MSVC_EMULATED)
#        define ALPAKA_NO_HOST_ACC_WARNING __pragma(hd_warning_disable)
#    else
#        define ALPAKA_NO_HOST_ACC_WARNING _Pragma("hd_warning_disable")
#    endif
#else
#    define ALPAKA_NO_HOST_ACC_WARNING
#endif

//-----------------------------------------------------------------------------
//! Macro defining the inline function attribute.
#if BOOST_LANG_CUDA || BOOST_LANG_HIP
#    define ALPAKA_FN_INLINE __forceinline__
#else
#    define ALPAKA_FN_INLINE inline
#endif

//-----------------------------------------------------------------------------
//! This macro defines a variable lying in global accelerator device memory.
//!
//! Example:
//!   ALPAKA_STATIC_ACC_MEM_GLOBAL int i;
//!
//! Those variables behave like ordinary variables when used in file-scope.
//! They have external linkage (are accessible from other compilation units).
//! If you want to access it from a different compilation unit, you have to declare it as extern:
//!   extern ALPAKA_STATIC_ACC_MEM_GLOBAL int i;
//! Like ordinary variables, only one definition is allowed (ODR)
//! Failure to do so might lead to linker errors.
//!
//! In contrast to ordinary variables, you can not define such variables
//! as static compilation unit local variables with internal linkage
//! because this is forbidden by CUDA.
#if(BOOST_LANG_CUDA && BOOST_ARCH_PTX) || (BOOST_LANG_HIP && (BOOST_ARCH_HSA || BOOST_ARCH_PTX))
#    define ALPAKA_STATIC_ACC_MEM_GLOBAL __device__
#else
#    define ALPAKA_STATIC_ACC_MEM_GLOBAL
#endif

//-----------------------------------------------------------------------------
//! This macro defines a variable lying in constant accelerator device memory.
//!
//! Example:
//!   ALPAKA_STATIC_ACC_MEM_CONSTANT int i;
//!
//! Those variables behave like ordinary variables when used in file-scope.
//! They have external linkage (are accessible from other compilation units).
//! If you want to access it from a different compilation unit, you have to declare it as extern:
//!   extern ALPAKA_STATIC_ACC_MEM_CONSTANT int i;
//! Like ordinary variables, only one definition is allowed (ODR)
//! Failure to do so might lead to linker errors.
//!
//! In contrast to ordinary variables, you can not define such variables
//! as static compilation unit local variables with internal linkage
//! because this is forbidden by CUDA.
#if(BOOST_LANG_CUDA && BOOST_ARCH_PTX) || (BOOST_LANG_HIP && (BOOST_ARCH_HSA || BOOST_ARCH_PTX))
#    define ALPAKA_STATIC_ACC_MEM_CONSTANT __constant__
#else
#    define ALPAKA_STATIC_ACC_MEM_CONSTANT
#endif
