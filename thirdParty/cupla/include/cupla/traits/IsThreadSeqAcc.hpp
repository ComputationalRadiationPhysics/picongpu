/* Copyright 2016 Rene Widera
 *
 * This file is part of cupla.
 *
 * cupla is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * cupla is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with cupla.
 * If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include "cupla/namespace.hpp"
#include "cupla/types.hpp"

namespace cupla
{
inline namespace CUPLA_ACCELERATOR_NAMESPACE
{
namespace traits
{

    /** check if thread level is full sequential
     *
     * \return ::value true if no threads where used in the thread level
     *                  else false
     */
    template< typename T_Acc >
    struct IsThreadSeqAcc
    {
        static constexpr bool value = false;
    };


#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED
    template<
        typename T_KernelDim,
        typename T_IndexType
    >
    struct IsThreadSeqAcc<
        ::alpaka::AccCpuOmp2Blocks<
            T_KernelDim,
            T_IndexType
        >
    >
    {
        static constexpr bool value = true;
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
    template<
        typename T_KernelDim,
        typename T_IndexType
    >
    struct IsThreadSeqAcc<
        ::alpaka::AccCpuSerial<
            T_KernelDim,
            T_IndexType
        >
    >
    {
        static constexpr bool value = true;
    };
#endif

#ifdef ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED
    template<
        typename T_KernelDim,
        typename T_IndexType
    >
    struct IsThreadSeqAcc<
        ::alpaka::AccCpuTbbBlocks<
            T_KernelDim,
            T_IndexType
        >
    >
    {
        static constexpr bool value = true;
    };
#endif

} // namespace traits
} // namespace CUPLA_ACCELERATOR_NAMESPACE
} // namespace cupla
