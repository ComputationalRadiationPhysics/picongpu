/**
 * Copyright 2015 Rene Widera, Alexander Grund
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


#include "types.h"
#include "nvidia/warp.hpp"
#include <boost/type_traits.hpp>
#include <math_functions.h>
#include <device_functions.h>
#include <climits>


namespace PMacc
{
namespace nvidia
{

    namespace detail {

        template<typename T_Type, bool T_isKepler>
        struct AtomicAllInc
        {
            DINLINE T_Type
            operator()(T_Type* ptr)
            {
                return atomicAdd(ptr, 1);
            }
        };

#if PMACC_CUDA_ARCH >= 300
       /**
         * Trait that returns whether an optimized version of AtomicAllInc
         * exists for Kepler architectures (and up)
         */
        template<typename T>
        struct AtomicAllIncIsOptimized
        {
            enum{
                value = boost::is_same<T,          int>::value ||
                        boost::is_same<T, unsigned int>::value ||
                        boost::is_same<T,          long long int>::value ||
                        boost::is_same<T, unsigned long long int>::value ||
                        boost::is_same<T, float>::value
            };
        };

        /**
         * AtomicAllInc for Kepler and up
         * Defaults to unoptimized version for unsupported types
         */
        template<typename T_Type, bool T_UseOptimized = AtomicAllIncIsOptimized<T_Type>::value>
        struct AtomicAllIncKepler: public AtomicAllInc<T_Type, false>
        {};

        /**
         * Optimized version
         *
         * This warp aggregated atomic increment implementation based on nvidia parallel forall example
         * http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
         * (author: Andrew Adinetz, date: October 1th, 2014)
         *
         */
        template<typename T_Type>
        struct AtomicAllIncKepler<T_Type, true>
        {
            DINLINE T_Type
            operator()(T_Type* ptr)
            {
                /* Get a bitmask with 1 for each thread in the warp, that executes this */
                const int mask = __ballot(1);
                /* select the leader */
                const int leader = __ffs(mask) - 1;
                T_Type result;
                const int laneId = getLaneId();
                /* Get the start value for this warp */
                if (laneId == leader)
                    result = atomicAdd(ptr, static_cast<T_Type>(__popc(mask)));
                result = warpBroadcast(result, leader);
                /* Add offset per thread */
                return result + static_cast<T_Type>(__popc(mask & ((1 << laneId) - 1)));
            }
        };

        /**
         * Optimized version for int64.
         * As CUDA atomicAdd does not support int64 directly we just cast it
         * and call the uint64 implementation
         */
        template<>
        struct AtomicAllIncKepler<long long int, true>
        {
            DINLINE long long int
            operator()(long long int* ptr)
            {
                return static_cast<long long int>(
                        AtomicAllIncKepler<unsigned long long int>()(
                                reinterpret_cast<unsigned long long int*>(ptr)
                        )
                );
            }
        };

        template<typename T_Type>
        struct AtomicAllInc<T_Type, true>: public AtomicAllIncKepler<T_Type>
        {};
#endif /* PMACC_CUDA_ARCH >= 300 */

    }  // namespace detail

/** optimized atomic increment
 *
 * - only optimized if PTX ISA >=3.0
 * - this atomic uses warp aggregation to speedup the operation compared to cuda `atomicInc()`
 * - cuda `atomicAdd()` is used if the compute architecture does not support warp aggregation
 * - all participate threads must change the same pointer (ptr) else the result is unspecified
 *
 * @param ptr pointer to memory (must be the same address for all threads in a block)
 *
 */
template<typename T>
DINLINE
T atomicAllInc(T *ptr)
{
    return detail::AtomicAllInc<T, (PMACC_CUDA_ARCH >= 300) >()(ptr);
}

/** optimized atomic value exchange
 *
 * - only optimized if PTX ISA >=2.0
 * - this atomic uses warp vote function to speedup the operation
 *   compared to cuda `atomicExch()`
 * - cuda `atomicExch()` is used if the compute architecture not supports
 *   warps vote functions
 * - all participate threads must change the same
 *   pointer (ptr) and set the same value, else the
 *   result is unspecified
 *
 * @param ptr pointer to memory (must be the same address for all threads in a block)
 * @param value new value (must be the same for all threads in a block)
 */
template<typename T_Type>
DINLINE void
atomicAllExch(T_Type* ptr, const T_Type value)
{
#if (__CUDA_ARCH__ >= 200)
    const int mask = __ballot(1);
    // select the leader
    const int leader = __ffs(mask) - 1;
    // leader does the update
    if (getLaneId() == leader)
#endif
        atomicExch(ptr, value);
}

} //namespace nvidia
} //namespace PMacc
