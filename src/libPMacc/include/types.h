/**
 * Copyright 2013 Felix Schmitt, Heiko Burau, René Widera, Wolfgang Hoenig
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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

#include <stdint.h>
#include <builtin_types.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdexcept>

#include "boost/typeof/std/utility.hpp"
#include "debug/PMaccVerbose.hpp"

#define BOOST_MPL_LIMIT_VECTOR_SIZE 20
#define BOOST_MPL_LIMIT_MAP_SIZE 20



#define PMACC_AUTO_TPL(var,...) BOOST_AUTO_TPL(var,(__VA_ARGS__))
#define PMACC_AUTO(var,...) BOOST_AUTO(var,(__VA_ARGS__))

namespace PMacc
{

//short name for access verbose types of libPMacc
typedef PMaccVerbose ggLog;

typedef uint64_t id_t;
typedef unsigned long long int uint64_cu;
typedef long long int int64_cu;

#define BOOST_MPL_LIMIT_VECTOR_SIZE 20

#define HDINLINE __device__ __host__ __forceinline__
#define DINLINE __device__ __forceinline__
#define HINLINE __host__ inline

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
#define PMACC_NO_NVCC_HDWARNING #pragma hd_warning_disable
#else
#define PMACC_NO_NVCC_HDWARNING
#endif

/**
 * Bitmask which describes the direction of communication.
 *
 * Bitmasks may be combined logically, e.g. LEFT+TOP = TOPLEFT.
 * It is not possible to combine complementary masks (e.g. FRONT and BACK),
 * as a bitmask always defines one direction of communication (send or receive).
 */
enum ExchangeType
{
    RIGHT = 1u, LEFT = 2u, BOTTOM = 3u, TOP = 6u, BACK = 9u, FRONT = 18u // 3er-System
};

#define MAX_EXCHANGE_TYPE FRONT + TOP + LEFT

/**
 * Defines number of dimensions (1-3)
 */
/*enum Dim
{
    DIM1 = 1u, DIM2 = 2u, DIM3 = 3u
};*/

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
 * Captures CUDA errors and prints messages to stdout, including line number and file.
 *
 * @param cmd command with cudaError_t return value to check
 */
#define CUDA_CHECK(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){std::cerr<<"<"<<__FILE__<<">:"<<__LINE__<<std::endl; throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error)));}}

#define CUDA_CHECK_MSG(cmd,msg) {cudaError_t error = cmd; if(error!=cudaSuccess){std::cerr<<"<"<<__FILE__<<">:"<<__LINE__<<msg<<std::endl; throw std::runtime_error(std::string("[CUDA] Error: ") + std::string(cudaGetErrorString(error)));}}

#define CUDA_CHECK_NO_EXCEP(cmd) {cudaError_t error = cmd; if(error!=cudaSuccess){printf("[CUDA] Error: <%s>:%i ",__FILE__,__LINE__);}}
/**
 * Converts an ExchangeType into a bitmask
 *
 * e.g. converts BOTTOM to "0000 0100"
 *
 * @param edge_type value from enum ExchangeType to convert
 */
#define EDGE_TO_MASK(edge_type) (1u<<(edge_type))

/**
 * Returns if edge_type is set in a bitmask
 *
 * e.g. mask="0000 1110", edge_type=BOTTOM, makro returns 2 (true)
 *
 * @param mask bitmask which represents an edge direction
 * @param edge_type Edge_Type to query
 * @return false (0) if ExchangeType bit is not set, true otherwise (!=0)
 */
#define IS_EDGE_SET(mask,edge_type) ((EDGE_TO_MASK((edge_type)))&(mask))

#define TO_BITS(x) (~((-1) << (x)))


/* calculate and set the optimal alignment for data
 * you must align all array and structs which can used on device
 * @param byte byte of data which must aligned
 */
#define __optimal_align__(byte)   \
        __align__(                \
        ((byte)==1?1:             \
        ((byte)<=2?2:             \
        ((byte)<=4?4:             \
        ((byte)<=8?8:             \
        ((byte)<=16?16:           \
        ((byte)<=32?32:           \
        ((byte)<=64?64:128        \
        ))))))))

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


#ifdef __CUDA_ARCH__ //we are on gpu
#define PMACC_PLACEHOLDER(id) using namespace PMACC_JOIN(device_placeholder,id)
#else
#define PMACC_PLACEHOLDER(id) using namespace PMACC_JOIN(host_placeholder,id)
#endif
/*define special makros for creating classes which are ony used as identifer*/
#define PMACC_identifier(in_type,name,in_default,id)                           \
    namespace PMACC_JOIN(placeholder_definition,id){                           \
        struct name{                                                           \
            typedef name ThisType;                                             \
            typedef in_type type;                                              \
            static const type defaultValue = in_default;                       \
        };                                                                     \
    }                                                                          \
    using namespace PMACC_JOIN(placeholder_definition,id);                     \
    namespace PMACC_JOIN(host_placeholder,id){                                 \
        PMACC_JOIN(placeholder_definition,id)::name PMACC_JOIN(name,_);        \
    }                                                                          \
    namespace PMACC_JOIN(device_placeholder,id){                               \
        __constant__ PMACC_JOIN(placeholder_definition,id)::name PMACC_JOIN(name,_);          \
    }                                                                          \
    PMACC_PLACEHOLDER(id);

#define identifier(in_type,name,in_default) PMACC_identifier(in_type,name,in_default,__COUNTER__)

/*define special makros for creating classes which are ony used as identifer*/
#define PMACC_wildcard(name,id,...)                                            \
    namespace PMACC_JOIN(placeholder_definition,id) {                          \
        struct name{                                                           \
            __VA_ARGS__                                                        \
        };                                                                     \
    }                                                                          \
    using namespace PMACC_JOIN(placeholder_definition,id);                     \
    namespace PMACC_JOIN(device_placeholder,id){                               \
        PMACC_JOIN(placeholder_definition,id)::name PMACC_JOIN(name,_);        \
    }                                                                          \
    namespace PMACC_JOIN(host_placeholder,id){                                 \
        __constant__ PMACC_JOIN(placeholder_definition,id)::name PMACC_JOIN(name,_);          \
    }                                                                          \
    PMACC_PLACEHOLDER(id);


/** create a wildcard (identifier with arbitrary code as second parameter
 * !! second parameter is optinal
 * example: wildcard(_1); //create name _1
 * example: wildcard(_1,typedef int type;); //create name _1, 
 *          later its possible: typedef _1::type type; 
 */
#define wildcard(name,...) PMACC_wildcard(name,__COUNTER__,__VA_ARGS__)


} //namespace PMacc

