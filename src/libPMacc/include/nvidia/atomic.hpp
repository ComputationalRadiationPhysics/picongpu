/**
 * Copyright 2015 Rene Widera
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
#include <math_functions.h>
#include <device_functions.h>


namespace PMacc
{
namespace nvidia
{

/** optimized atomic increment
 *
 * - only optimized if PTX ISA >=3.0
 * - this atomic uses warp aggregation to speedup the operation compared to
 *   cuda `atomicInc()`
 * - cuda `atomicAdd()` is used if the compute architecture not supports
 *   warp aggregation
 *   is used
 * - all participate threads must change the same
 *   pointer (ptr) else the result is unspecified
 *
 * @param ptr pointer to memory (must be the same address for all threads in a block)
 *
 * This warp aggregated atomic increment implementation based on
 * nvidia parallel forall example
 * http://devblogs.nvidia.com/parallelforall/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
 */
DINLINE
int atomicAllInc(int *ptr)
{
#if (__CUDA_ARCH__ >= 300)
    const int mask = __ballot(1);
    /* select the leader */
    const int leader = __ffs(mask) - 1;
    int restult;
    const int lanId = getLaneId();
    if (lanId == leader)
        restult = atomicAdd(ptr, __popc(mask));
    restult = warpBroadcast(restult, leader);
    /* each thread computes its own value */
    return restult + __popc(mask & ((1 << lanId) - 1));
#else
    return atomicAdd(ptr,1);
#endif
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
