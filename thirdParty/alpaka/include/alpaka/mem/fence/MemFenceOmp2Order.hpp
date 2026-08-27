/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Config.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/Unreachable.hpp"
#include "alpaka/mem/order/MemoryOrder.hpp"

#include <atomic>
#include <concepts>

#if defined(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)

#    if ALPAKA_OMP < ALPAKA_VERSION_NUMBER(2002, 03, 0)
#        ifdef(ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED)
#            error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#        endif
#        ifdef(ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED)
#            error If ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#        endif
#    endif

namespace alpaka::detail
{

    template<MemoryOrder TMemOrder>
    inline auto flushOmp(TMemOrder)
    {
        if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
        {
            // sequenital consistency in openMP is only enforced in implicit flush.
            // We use std atomics since our openMP is limited to CPU backends
            std::atomic_thread_fence(std::memory_order::seq_cst);
        }
        else if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
        {
            // Flush orderings were introduced in OpenMP 5.0
#    if ALPAKA_OMP >= ALPAKA_VERSION_NUMBER(2018, 11, 0)
#        pragma omp flush acq_rel
#    else
#        pragma omp flush
#    endif
        }
        else if constexpr(std::same_as<TMemOrder, mem_order::Release>)
        {
            // Flush orderings were introduced in OpenMP 5.0
#    if ALPAKA_OMP >= ALPAKA_VERSION_NUMBER(2018, 11, 0)
#        pragma omp flush release
#    else
#        pragma omp flush
#    endif
        }
        else if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
        {
            // Flush orderings were introduced in OpenMP 5.0
#    if ALPAKA_OMP >= ALPAKA_VERSION_NUMBER(2018, 11, 0)
#        pragma omp flush acquire
#    else
#        pragma omp flush
#    endif
        }
        else if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
        {
            // Relaxed memory barrier is a no op
        }
        else
        {
            ALPAKA_UNREACHABLE();
        }
    }
} // namespace alpaka::detail

#endif
