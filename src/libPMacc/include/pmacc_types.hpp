/**
 * Copyright 2013-2016 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz,
 *                     Alexander Grund
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "debug/PMaccVerbose.hpp"
#include "ppFunctions.hpp"

#define BOOST_MPL_LIMIT_VECTOR_SIZE 20
#define BOOST_MPL_LIMIT_MAP_SIZE 20
#include <boost/typeof/std/utility.hpp>
#include <boost/mpl/placeholders.hpp>
#include <boost/filesystem.hpp>

// compatibility macros (compiler or C++ standard version specific)
#include <boost/config.hpp>

#include <builtin_types.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <stdint.h>
#include <stdexcept>

#define PMACC_AUTO_TPL(var,...) BOOST_AUTO_TPL(var,(__VA_ARGS__))
#define PMACC_AUTO(var,...) BOOST_AUTO(var,(__VA_ARGS__))

namespace PMacc
{

namespace bmpl = boost::mpl;
namespace bfs = boost::filesystem;

//short name for access verbose types of libPMacc
typedef PMaccVerbose ggLog;

typedef uint64_t id_t;
typedef unsigned long long int uint64_cu;
typedef long long int int64_cu;

#define HDINLINE __device__ __host__ __forceinline__
#define DINLINE __device__ __forceinline__
#define HINLINE __host__ inline

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
#define PMACC_GLOBAL_KEYWORD __location__(global)

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
    __align__(                                                                 \
        /** \bug avoid bug if alignment is >16 byte                            \
         * https://github.com/ComputationalRadiationPhysics/picongpu/issues/1563 \
         */                                                                    \
        PMACC_MIN(PMACC_ROUND_UP_NEXT_POW2(byte),16)                           \
    )

#define PMACC_ALIGN(var,...) __optimal_align__(sizeof(__VA_ARGS__)) __VA_ARGS__ var
#define PMACC_ALIGN8(var,...) __align__(8) __VA_ARGS__ var

/*! area which is calculated
 *
 * CORE is the inner area of a grid
 * BORDER is the border of a grid (my own border, not the neighbor part)
 */
enum AreaType
{
    CORE = 1u, BORDER = 2u, GUARD = 4u
};

#define __delete(var) if((var)) { delete (var); var=NULL; }
#define __deleteArray(var) if((var)) { delete[] (var); var=NULL; }

} //namespace PMacc
