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
#include <device_functions.h>


namespace PMacc
{
namespace nvidia
{

/** get lane id of a thread within a warp
 *
 * id is in range [0,WAPRSIZE-1]
 * required PTX ISA >=1.3
 */
#if (__CUDA_ARCH__ >= 130)
DINLINE uint32_t getLaneId()
{
    uint32_t id;
    asm("mov.u32 %0, %laneid;" : "=r" (id));
    return id;
}
#endif

/** broadcast data within a thread
 *
 * required PTX ISA >=3.0
 */
#if (__CUDA_ARCH__ >= 300)
DINLINE int32_t warpBroadcast(const int32_t data, const int32_t srcLaneId)
{
    return  __shfl(data, srcLaneId);
}
/**
 * Broadcast a 64bit integer by using 2 32bit broadcasts
 */
DINLINE int64_cu warpBroadcast(const int64_cu data, const int32_t srcLaneId)
{
    union{
        int32_t dataHi, dataLo;
        int64_cu data64;
    } dataUnion;
    dataUnion.data64 = data;
    warpBroadcast(dataUnion.dataHi, srcLaneId);
    warpBroadcast(dataUnion.dataLo, srcLaneId);
    return dataUnion.data64;
}
/**
 * Broadcast a 32bit unsigned int
 * Maps to signed int function with no additional overhead
 */
DINLINE uint32_t warpBroadcast(const uint32_t data, const int32_t srcLaneId)
{
    return static_cast<uint32_t>(
            warpBroadcast(static_cast<int32_t>(data), srcLaneId)
            );
}
/**
 * Broadcast a 64bit unsigned int
 * Maps to signed int function with no additional overhead
 */
DINLINE uint64_cu warpBroadcast(const uint64_cu data, const int32_t srcLaneId)
{
    return static_cast<uint64_cu>(
            warpBroadcast(static_cast<uint64_cu>(data), srcLaneId)
            );
}

/**
 * Broadcast a 32bit float
 */
DINLINE float warpBroadcast(const float data, const int32_t srcLaneId)
{
    return  __shfl(data, srcLaneId);
}
/**
 * Broadcast a 64bit float by using 2 32bit broadcasts
 */
DINLINE double warpBroadcast(const double data, const int32_t srcLaneId)
{
    union{
        float dataHi, dataLo;
        double data64;
    } dataUnion;
    dataUnion.data64 = data;
    warpBroadcast(dataUnion.dataHi, srcLaneId);
    warpBroadcast(dataUnion.dataLo, srcLaneId);
    return dataUnion.data64;
}
#endif

} //namespace nvidia
} //namespace PMacc
