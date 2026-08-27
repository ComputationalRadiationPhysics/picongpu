/* Copyright 2022 Jan Stephan, Andrea Bocci, Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Interface.hpp"
#include "alpaka/mem/fence/Traits.hpp"
#include "alpaka/mem/order/MemoryOrderStl.hpp"

#include <atomic>

namespace alpaka
{
    //! The serial CPU memory fence.
    class MemFenceCpuSerial : public interface::Implements<ConceptMemFence, MemFenceCpuSerial>
    {
    };

    namespace trait
    {
        template<>
        struct MemFenceDefaultOrder<MemFenceCpuSerial>
        {
            using type = mem_order::AcqRel;
            static constexpr auto value = mem_order::acq_rel;
        };

        template<MemoryOrder TMemOrder>
        struct MemFence<MemFenceCpuSerial, TMemOrder, memory_scope::Block>
        {
            static auto mem_fence(MemFenceCpuSerial const&, TMemOrder, memory_scope::Block const&)
            {
                /* Nothing to be done on the block level for the serial case. */
            }
        };

        template<MemoryOrder TMemOrder>
        struct MemFence<MemFenceCpuSerial, TMemOrder, memory_scope::Grid>
        {
            static auto mem_fence(MemFenceCpuSerial const&, TMemOrder, memory_scope::Grid const&)
            {
                /* Nothing to be done on the grid level for the serial case. */
            }
        };

        template<MemoryOrder TMemOrder, MemoryScope TMemScope>
        struct MemFence<MemFenceCpuSerial, TMemOrder, TMemScope>
        {
            static auto mem_fence(MemFenceCpuSerial const&, TMemOrder order, TMemScope const&)
            {
                /* Enable device fences because we may want to synchronize with other (serial) kernels. */
                std::atomic_thread_fence(MemOrderStl::get(order));
            }
        };
    } // namespace trait
} // namespace alpaka
