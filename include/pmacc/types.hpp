/* Copyright 2013-2018 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <cupla/types.hpp>

#ifndef PMACC_CUDA_ENABLED
#   define PMACC_CUDA_ENABLED ALPAKA_ACC_GPU_CUDA_ENABLED
#endif

#if( PMACC_CUDA_ENABLED == 1 )
/* include mallocMC before cupla renaming is activated, else we need the variable acc
 * to call atomic cuda functions
 */
#   include <mallocMC/mallocMC.hpp>
#endif


#include <cuda_to_cupla.hpp>

#if( PMACC_CUDA_ENABLED == 1 )
/** @todo please remove this workaround
 * This workaround allows to use native CUDA on the CUDA device without
 * passing the variable `acc` to each function. This is only needed during the
 * porting phase to allow the full feature set of the plain PMacc and PIConGPU
 * CUDA version if the accelerator is CUDA.
 */
#   undef blockIdx
#   undef __syncthreads
#   undef threadIdx
#   undef gridDim
#   undef blockDim
#   undef uint3
#   undef dim3

#endif

#include "pmacc/debug/PMaccVerbose.hpp"
#include "pmacc/ppFunctions.hpp"

#define BOOST_MPL_LIMIT_VECTOR_SIZE 20
#define BOOST_MPL_LIMIT_MAP_SIZE 20
#include <boost/typeof/std/utility.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/filesystem.hpp>

// compatibility macros (compiler or C++ standard version specific)
#include <boost/config.hpp>

#include <stdint.h>
#include <stdexcept>


namespace pmacc
{

namespace bmpl = boost::mpl;
namespace bfs = boost::filesystem;

//short name for access verbose types of PMacc
typedef PMaccVerbose ggLog;

typedef uint64_t id_t;
typedef unsigned long long int uint64_cu;
typedef long long int int64_cu;

#define HDINLINE ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE
#define DINLINE ALPAKA_FN_ACC ALPAKA_FN_INLINE
#define DEVICEONLY ALPAKA_FN_ACC
#define HINLINE ALPAKA_FN_HOST ALPAKA_FN_INLINE

/**
 * CUDA architecture version (aka PTX ISA level)
 * 0 for host compilation
 */
#ifndef __CUDA_ARCH__
#   define PMACC_CUDA_ARCH 0
#else
#   define PMACC_CUDA_ARCH __CUDA_ARCH__
#endif

/** PMacc global identifier for CUDA kernel */
#define PMACC_GLOBAL_KEYWORD DINLINE

/*
 * Disable nvcc warning:
 * calling a __host__ function from __host__ __device__ function.
 *
 * Usage:
 * PMACC_NO_NVCC_HDWARNING
 * HDINLINE function_declaration()
 *
 * It is not possible to disable the warning for a __host__ function
 * if there are calls of virtual functions inside. For this case use a wrapper
 * function.
 * WARNING: only use this method if there is no other way to create runable code.
 * Most cases can solved by #ifdef __CUDA_ARCH__ or #ifdef __CUDACC__.
 */
#if defined(__CUDACC__)
#define PMACC_NO_NVCC_HDWARNING _Pragma("hd_warning_disable")
#else
#define PMACC_NO_NVCC_HDWARNING
#endif

/**
 * Bitmask which describes the direction of communication.
 *
 * Bitmasks may be combined logically, e.g. LEFT+TOP = TOPLEFT.
 * It is not possible to combine complementary masks (e.g. FRONT and BACK),
 * as a bitmask always defines one direction of communication (send or receive).
 *
 * Axis index relation:
 *   right & left are in X
 *   bottom & top are in Y
 *   back & front are in Z
 */
enum ExchangeType
{
    RIGHT = 1u, LEFT = 2u, BOTTOM = 3u, TOP = 6u, BACK = 9u, FRONT = 18u // 3er-System
};

struct ExchangeTypeNames
{
    std::string operator[]( const uint32_t exchange ) const
    {
        const char* names[27] = {
            "none",
            "right", "left", "bottom",
            "right-bottom", "left-bottom",
            "top",
            "right-top", "left-top",
            "back",
            "right-back", "left-back",
            "bottom-back", "right-bottom-back", "left-bottom-back",
            "top-back", "right-top-back", "left-top-back",
            "front",
            "right-front", "left-front",
            "bottom-front", "right-bottom-front", "left-bottom-front",
            "top-front", "right-top-front", "left-top-front"
        };
        return names[exchange];
    }
};

/**
 * Defines number of dimensions (1-3)
 */

#define DIM1 1u
#define DIM2 2u
#define DIM3 3u

/**
 * Internal event/task type used for notifications in the event system.
 */
enum EventType
{
    FINISHED, COPYHOST2DEVICE, COPYDEVICE2HOST, COPYDEVICE2DEVICE, SENDFINISHED, RECVFINISHED, LOGICALAND, SETVALUE, GETVALUE, KERNEL
};

/**
 * Print a cuda error message including file/line info to stderr
 */
#define PMACC_PRINT_CUDA_ERROR(msg) \
    std::cerr << "[CUDA] Error: <" << __FILE__ << ">:" << __LINE__ << " " << msg << std::endl

/**
 * Print a cuda error message including file/line info to stderr and raises an exception
 */
#define PMACC_PRINT_CUDA_ERROR_AND_THROW(cudaError, msg) \
    PMACC_PRINT_CUDA_ERROR(msg);                         \
    throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(cudaError)))

/**
 * Captures CUDA errors and prints messages to stdout, including line number and file.
 *
 * @param cmd command with cudaError_t return value to check
 */
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){ PMACC_PRINT_CUDA_ERROR_AND_THROW(error, ""); }}

#define CUDA_CHECK_MSG(cmd,msg) {cudaError_t error = cmd; if(error!=cudaSuccess){ PMACC_PRINT_CUDA_ERROR_AND_THROW(error, msg); }}

#define CUDA_CHECK_NO_EXCEP(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){ PMACC_PRINT_CUDA_ERROR(""); }}

/** calculate and set the optimal alignment for data
  *
  * you must align all arrays and structs that are used on the device
  * @param byte size of data in bytes
  */
#define __optimal_align__(byte)                                                \
    alignas(                                                                 \
        /** \bug avoid bug if alignment is >16 byte                            \
         * https://github.com/ComputationalRadiationPhysics/picongpu/issues/1563 \
         */                                                                    \
        PMACC_MIN(PMACC_ROUND_UP_NEXT_POW2(byte),16)                           \
    )

#define PMACC_ALIGN(var,...) __optimal_align__(sizeof(__VA_ARGS__)) __VA_ARGS__ var
#define PMACC_ALIGN8( var, ... ) alignas( 8 ) __VA_ARGS__ var

/*! area which is calculated
 *
 * CORE is the inner area of a grid
 * BORDER is the border of a grid (my own border, not the neighbor part)
 */
enum AreaType
{
    CORE = 1u, BORDER = 2u, GUARD = 4u
};

#define __delete(var) if((var)) { delete (var); var=nullptr; }
#define __deleteArray(var) if((var)) { delete[] (var); var=nullptr; }

/**
 * Visual Studio has a bug with constexpr variables being captured in lambdas as
 * non-constexpr variables, causing build errors. The issue has been verified
 * for versions 14.0 and 15.5 (latest at the moment) and is also reported in
 * https://stackoverflow.com/questions/28763375/using-lambda-captured-constexpr-value-as-an-array-dimension
 * and related issue
 * https://developercommunity.visualstudio.com/content/problem/1997/constexpr-not-implicitely-captured-in-lambdas.html
 *
 * As a workaround (until this is fixed in VS) add a new PMACC_CONSTEXPR_CAPTURE
 * macro for declaring constexpr variables that are captured in lambdas and have
 * to remain constexpr inside a lambda e.g., used as a template argument. Such
 * variables have to be declared with PMACC_CONSTEXPR_CAPTURE instead of
 * constexpr. The macro will be replaced with just constexpr for other compilers
 * and for Visual Studio with static constexpr, which makes it capture properly.
 *
 * Note that this macro is to be used only in very few cases, where not only a
 * constexpr is captured, but also it has to remain constexpr inside a lambda.
 */
#ifdef _MSC_VER
#   define PMACC_CONSTEXPR_CAPTURE static constexpr
#else
#   define PMACC_CONSTEXPR_CAPTURE constexpr
#endif

} //namespace pmacc
