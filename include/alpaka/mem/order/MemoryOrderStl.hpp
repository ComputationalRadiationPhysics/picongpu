/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/order/MemoryOrder.hpp"

#include <atomic>
#include <concepts>

namespace alpaka
{
    struct MemOrderStl
    {
        template<MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder)
        {
            if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
            {
                return std::memory_order::seq_cst;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
            {
                return std::memory_order::acq_rel;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Release>)
            {
                return std::memory_order::release;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
            {
                return std::memory_order::acquire;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
            {
                return std::memory_order::relaxed;
            }
        }
    };

} // namespace alpaka
