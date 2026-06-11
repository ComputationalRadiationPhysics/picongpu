/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
        template<uint32_t T_maxWorkers, typename T_Acc = Acc<DIM1>>
        struct GetNumWorkers
        {
            static constexpr uint32_t value = T_maxWorkers;
        };

#if (ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
        template<uint32_t T_maxWorkers, typename... T_Args>
        struct GetNumWorkers<T_maxWorkers, alpaka::AccCpuOmp2Blocks<T_Args...>>
        {
            static constexpr uint32_t value = 1u;
        };
#endif
#if (ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
        template<uint32_t T_maxWorkers, typename... T_Args>
        struct GetNumWorkers<T_maxWorkers, alpaka::AccCpuSerial<T_Args...>>
        {
            static constexpr uint32_t value = 1u;
        };
#endif
#if (ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
        template<uint32_t T_maxWorkers, typename... T_Args>
        struct GetNumWorkers<T_maxWorkers, alpaka::AccCpuTbbBlocks<T_Args...>>
        {
            static constexpr uint32_t value = 1u;
        };
#endif
    } // namespace traits
} // namespace pmacc
