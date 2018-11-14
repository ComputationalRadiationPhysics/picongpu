/**
 * \file
 * Copyright 2017-2018 Alexander Matthes, Benjamin Worpitz
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

#include <boost/predef.h>

//-----------------------------------------------------------------------------
// In boost since 1.68.0
// BOOST_PREDEF_MAKE_10_VVRRP(V)
#if !defined(BOOST_PREDEF_MAKE_10_VVRRP)
    #define BOOST_PREDEF_MAKE_10_VVRRP(V) BOOST_VERSION_NUMBER(((V)/1000)%100,((V)/10)%100,(V)%10)
#endif

//---------------------------------------HIP-----------------------------------
// __HIPCC__ is defined by hipcc (if either __HCC__ or __CUDACC__ is defined)
#if !defined(BOOST_LANG_HIP)
  #if defined(__HIPCC__) && ( defined(__CUDACC__) || defined(__HCC__) )
    #include <hip/hip_runtime.h>
    //HIP defines "abort()" as "{asm("trap;");}", which breaks some kernels
    #undef abort
    // there is no HIP_VERSION macro
    #define BOOST_LANG_HIP BOOST_VERSION_NUMBER_AVAILABLE
    #if defined(BOOST_LANG_CUDA) && BOOST_LANG_CUDA
        #undef BOOST_LANG_CUDA
        #define BOOST_LANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
    #endif
  #else
    #define BOOST_LANG_HIP BOOST_VERSION_NUMBER_NOT_AVAILABLE
  #endif
#endif

//-----------------------------------------------------------------------------
// HSA device architecture detection (HSA generated via HIP(HCC) or HCC directly)
#if !defined(BOOST_ARCH_HSA)
    #if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__==1 && defined(__HCC__) \
        || (defined(__HCC_ACCELERATOR__) && __HCC_ACCELERATOR__!=0)
        // __HIP_DEVICE_COMPILE__ does not represent feature capability of target device like CUDA_ARCH.
        // For feature detection there are special macros, see ROCm's HIP porting guide.
        #define BOOST_ARCH_HSA BOOST_VERSION_NUMBER_AVAILABLE
    #else
        #define BOOST_ARCH_HSA BOOST_VERSION_NUMBER_NOT_AVAILABLE
    #endif
#endif

//-----------------------------------------------------------------------------
// hcc HSA compiler detection
#if !defined(BOOST_COMP_HCC)
    #if defined(__HCC__)
        #define BOOST_COMP_HCC BOOST_VERSION_NUMBER_AVAILABLE
    #else
        #define BOOST_COMP_HCC BOOST_VERSION_NUMBER_NOT_AVAILABLE
    #endif
#endif

//-----------------------------------------------------------------------------
// In boost since 1.68.0
// CUDA language detection
// - clang defines __CUDA__ and __CUDACC__ when compiling CUDA code ('-x cuda')
// - nvcc defines __CUDACC__ when compiling CUDA code
#if !defined(BOOST_LANG_CUDA)
    #if defined(__CUDA__) || defined(__CUDACC__)
        #include <cuda.h>
        #define BOOST_LANG_CUDA BOOST_PREDEF_MAKE_10_VVRRP(CUDA_VERSION)
    #else
        #define BOOST_LANG_CUDA BOOST_VERSION_NUMBER_NOT_AVAILABLE
    #endif
#endif

//-----------------------------------------------------------------------------
// In boost since 1.68.0
// CUDA device architecture detection
#if !defined(BOOST_ARCH_PTX)
    #if defined(__CUDA_ARCH__)
        #define BOOST_ARCH_PTX BOOST_PREDEF_MAKE_10_VRP(__CUDA_ARCH__)
    #else
        #define BOOST_ARCH_PTX BOOST_VERSION_NUMBER_NOT_AVAILABLE
    #endif
#endif

//-----------------------------------------------------------------------------
// In boost since 1.68.0
// nvcc CUDA compiler detection

#include <boost/version.hpp>
#if BOOST_VERSION >= 106800
    // BOOST_COMP_NVCC_EMULATED is defined by boost instead of BOOST_COMP_NVCC
    #if defined(BOOST_COMP_NVCC) && defined(BOOST_COMP_NVCC_EMULATED)
        #undef BOOST_COMP_NVCC
        #define BOOST_COMP_NVCC BOOST_COMP_NVCC_EMULATED
    #endif
#endif

#if !defined(BOOST_COMP_NVCC)
    #if defined(__NVCC__)
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
#endif

//-----------------------------------------------------------------------------
// In boost since 1.64.0
// Work around for broken intel detection
#if BOOST_COMP_INTEL == 0
    #if defined(__INTEL_COMPILER)
        #ifdef BOOST_COMP_INTEL_DETECTION
            #undef BOOST_COMP_INTEL_DETECTION
        #endif
        #define BOOST_COMP_INTEL_DETECTION BOOST_PREDEF_MAKE_10_VVRR(__INTEL_COMPILER)
        #if defined(BOOST_COMP_INTEL)
            #undef BOOST_COMP_INTEL
        #endif
        #define BOOST_COMP_INTEL BOOST_COMP_INTEL_DETECTION
    #endif
#endif
