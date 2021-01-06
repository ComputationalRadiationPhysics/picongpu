/* Copyright 2015-2021 Rene Widera, Alexander Grund
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

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)

#    include "pmacc/types.hpp"


namespace pmacc
{
    namespace nvidia
    {
/** get lane id of a thread within a warp
 *
 * id is in range [0,WAPRSIZE-1]
 * required PTX ISA >=1.3
 */
#    if(__CUDA_ARCH__ >= 130)
        DINLINE uint32_t getLaneId()
        {
            uint32_t id;
            asm("mov.u32 %0, %%laneid;" : "=r"(id));
            return id;
        }
#    elif BOOST_COMP_HIP
        DINLINE uint32_t getLaneId()
        {
            return __lane_id();
        }
#    endif


#    if(__CUDA_ARCH__ >= 300 || BOOST_COMP_HIP)
        /** broadcast data within a warp
         *
         * required PTX ISA >=3.0
         *
         * @param data value to broadcast
         * @param srcLaneId lane id of the source thread
         * @return value send by the source thread
         *
         * \{
         */
        //! broadcast a 32bit integer
        DINLINE int32_t warpBroadcast(const int32_t data, const int32_t srcLaneId)
        {
#        if(__CUDACC_VER_MAJOR__ >= 9)
            return __shfl_sync(__activemask(), data, srcLaneId);
#        else
            return __shfl(data, srcLaneId);
#        endif
        }

        //! Broadcast a 64bit integer by using 2 32bit broadcasts
        DINLINE int64_cu warpBroadcast(int64_cu data, const int32_t srcLaneId)
        {
            int32_t* const pData = reinterpret_cast<int32_t*>(&data);
            pData[0] = warpBroadcast(pData[0], srcLaneId);
            pData[1] = warpBroadcast(pData[1], srcLaneId);
            return data;
        }

        //! Broadcast a 32bit unsigned int
        DINLINE uint32_t warpBroadcast(const uint32_t data, const int32_t srcLaneId)
        {
            return static_cast<uint32_t>(warpBroadcast(static_cast<int32_t>(data), srcLaneId));
        }

        //! Broadcast a 64bit unsigned int
        DINLINE uint64_cu warpBroadcast(const uint64_cu data, const int32_t srcLaneId)
        {
            return static_cast<uint64_cu>(warpBroadcast(static_cast<int64_cu>(data), srcLaneId));
        }

        //! Broadcast a 32bit float
        DINLINE float warpBroadcast(const float data, const int32_t srcLaneId)
        {
#        if(__CUDACC_VER_MAJOR__ >= 9)
            return __shfl_sync(__activemask(), data, srcLaneId);
#        else
            return __shfl(data, srcLaneId);
#        endif
        }

        //! Broadcast a 64bit float by using 2 32bit broadcasts
        DINLINE double warpBroadcast(double data, const int32_t srcLaneId)
        {
            float* const pData = reinterpret_cast<float*>(&data);
            pData[0] = warpBroadcast(pData[0], srcLaneId);
            pData[1] = warpBroadcast(pData[1], srcLaneId);
            return data;
        }
//! @}
#    endif

    } // namespace nvidia
} // namespace pmacc

#endif
