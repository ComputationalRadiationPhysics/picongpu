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
DINLINE int warpBroadcast(const int data, const int srcLaneId)
{
    return  __shfl(data, srcLaneId);
}
#endif

} //namespace nvidia
} //namespace PMacc
