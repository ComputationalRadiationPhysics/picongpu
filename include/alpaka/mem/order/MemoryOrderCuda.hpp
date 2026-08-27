/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Config.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/mem/order/MemoryOrder.hpp"

#include <concepts>

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
namespace alpaka
{
    struct MemOrderCuda
    {
        template<MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder)
        {
#    ifdef ALPAKA_CUDA_ATOMIC
            if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
            {
                return ::cuda::memory_order_seq_cst;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
            {
                return ::cuda::memory_order_acq_rel;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Release>)
            {
                return ::cuda::memory_order_release;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
            {
                return ::cuda::memory_order_acquire;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
            {
                return ::cuda::memory_order_relaxed;
            }
#    else
#        if ALPAKA_LANG_CUDA >= ALPAKA_VERSION_NUMBER(12, 8, 0) && ALPAKA_ARCH_PTX

            if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
            {
                return __NV_ATOMIC_SEQ_CST;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
            {
                return __NV_ATOMIC_ACQ_REL;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Release>)
            {
                return __NV_ATOMIC_RELEASE;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
            {
                return __NV_ATOMIC_ACQUIRE;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
            {
                return __NV_ATOMIC_RELAXED;
            }
#        endif
#    endif
        }
    };

} // namespace alpaka

#endif
