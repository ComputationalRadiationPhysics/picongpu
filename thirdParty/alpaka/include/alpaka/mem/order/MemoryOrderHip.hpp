/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/order/MemoryOrder.hpp"

#include <concepts>

#ifdef ALPAKA_ACC_GPU_HIP_ENABLED

namespace alpaka
{
    struct MemOrderHip
    {
        template<MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder)
        {
            if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
            {
                return __ATOMIC_SEQ_CST;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
            {
                return __ATOMIC_ACQ_REL;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Release>)
            {
                return __ATOMIC_RELEASE;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
            {
                return __ATOMIC_ACQUIRE;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
            {
                return __ATOMIC_RELAXED;
            }
        }
    };

} // namespace alpaka

#endif
