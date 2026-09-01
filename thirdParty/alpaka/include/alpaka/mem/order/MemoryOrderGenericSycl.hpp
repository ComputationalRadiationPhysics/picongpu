/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/order/MemoryOrder.hpp"

#include <concepts>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{

    struct MemOrderSycl
    {
        template<MemoryOrder TMemOrder>
        static constexpr auto get(TMemOrder)
        {
            if constexpr(std::same_as<TMemOrder, mem_order::SeqCst>)
            {
                return sycl::memory_order::seq_cst;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::AcqRel>)
            {
                return sycl::memory_order::acq_rel;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Release>)
            {
                return sycl::memory_order::release;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Acquire>)
            {
                return sycl::memory_order::acquire;
            }
            if constexpr(std::same_as<TMemOrder, mem_order::Relaxed>)
            {
                return sycl::memory_order::relaxed;
            }
        }
    };

} // namespace alpaka

#endif
