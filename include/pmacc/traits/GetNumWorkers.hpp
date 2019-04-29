/* Copyright 2017-2019 Rene Widera
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

#include "pmacc/types.hpp"
#include <type_traits>


namespace pmacc
{
namespace traits
{
    /** Get number of workers
     *
     * the number of workers for a kernel depending on the used accelerator
     *
     * @tparam T_maxWorkers the maximum number of workers
     * @tparam T_Acc the accelerator type
     * @return @p ::value number of workers
     */
    template<
        uint32_t T_maxWorkers,
        typename T_Acc = cupla::AccThreadSeq
    >
    struct GetNumWorkers
    {
        static constexpr uint32_t value = T_maxWorkers;
    };

#if( ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED == 1 )
    template<
        uint32_t T_maxWorkers,
        typename ... T_Args
    >
    struct GetNumWorkers<
        T_maxWorkers,
        alpaka::acc::AccCpuOmp2Blocks< T_Args... >
    >
    {
        static constexpr uint32_t value = 1u;
    };
#endif
#if( ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED == 1 )
    template<
        uint32_t T_maxWorkers,
        typename ... T_Args
    >
    struct GetNumWorkers<
        T_maxWorkers,
        alpaka::acc::AccCpuSerial< T_Args... >
    >
    {
        static constexpr uint32_t value = 1u;
    };
#endif
#if( ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED == 1 )
    template<
        uint32_t T_maxWorkers,
        typename ... T_Args
    >
    struct GetNumWorkers<
        T_maxWorkers,
        alpaka::acc::AccCpuTbbBlocks< T_Args... >
    >
    {
        static constexpr uint32_t value = 1u;
    };
#endif
} // namespace traits
} // namespace pmacc
