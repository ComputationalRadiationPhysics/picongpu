/* Copyright 2015-2023 Rene Widera, Alexander Grund
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

#include <alpaka/warp/Traits.hpp>

#if(BOOST_LANG_CUDA || BOOST_COMP_HIP)

#    include "pmacc/types.hpp"


namespace pmacc
{
    namespace kernel
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

        /** broadcast data within a warp without using shared memory
         *
         * @param mask mask to select threads participating in the operation
         * @param data value to broadcast
         * @param srcLaneId lane id of the source thread
         * @return value send by the source thread
         */
        template<typename T_MaskType, typename T_DataType>
        DINLINE T_DataType warpBroadcast(T_MaskType const mask, T_DataType const data, int const srcLaneId);

        /** check if T is equal to at least one type in the tuple list
         *
         * @param tuple of types, each type is checked against T
         * @return true if T is equal to at least one type from the tuple list, else false
         */
        template<typename T, typename... T_TypeList>
        inline constexpr bool matchAnyType(std::tuple<T_TypeList...>)
        {
            return std::disjunction_v<std::is_same<T, T_TypeList>...>;
        }


        /** fallback for not natively supported types
         *
         * emulates the broadcast by repeating 32bit warp broadcasts
         */
        template<typename T_MaskType, typename T_DataType, typename T_Sfinae = void>
        struct WarpBroadcast
        {
            DINLINE T_DataType operator()(T_MaskType const mask, T_DataType data, int const srcLaneId) const
            {
                static_assert(sizeof(T_DataType) >= sizeof(int) && (sizeof(T_DataType) % sizeof(int)) == 0);
                int* const pData = reinterpret_cast<int*>(&data);
                for(int i = 0; i < sizeof(T_DataType) / sizeof(int); ++i)
                    pData[i] = warpBroadcast(mask, pData[i], srcLaneId);

                return data;
            }
        };

        /** List of types natively supported by CUDAs and HIPs shfl function interface */
        using VendorSupportedTypesForShfl = std::tuple<
            int,
            unsigned,
            float
        /* clang as cuda compiler does not support 64 bit warp shfl.
         * For AMD GPUs 64bit shufl will be provided by HIP.
         */
#        if(!BOOST_COMP_CLANG_CUDA)
            ,
            double,
            unsigned long,
            unsigned long long,
            long,
            long long
#        endif
            >;

        /** CUDA and HIP native supported data types */
        template<typename T_MaskType, typename T_DataType>
        struct WarpBroadcast<
            T_MaskType,
            T_DataType,
            std::enable_if_t<matchAnyType<T_DataType>(VendorSupportedTypesForShfl{})>>
        {
            DINLINE T_DataType operator()(T_MaskType const mask, T_DataType const data, int const srcLaneId) const
            {
#        if(BOOST_COMP_HIP)
                return __shfl(data, srcLaneId);
#        else
                return __shfl_sync(mask, data, srcLaneId);
#        endif
            }
        };

        template<typename T_MaskType, typename T_DataType>
        DINLINE T_DataType warpBroadcast(T_MaskType const mask, T_DataType const data, int const srcLaneId)
        {
            return WarpBroadcast<T_MaskType, T_DataType>{}(mask, data, srcLaneId);
        }
#    endif

    } // namespace kernel
} // namespace pmacc

#endif
