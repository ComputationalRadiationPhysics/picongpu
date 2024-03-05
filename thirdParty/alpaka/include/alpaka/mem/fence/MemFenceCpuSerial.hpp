/* Copyright 2022 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/mem/fence/Traits.hpp"

#include <atomic>

namespace alpaka
{
    //! The serial CPU memory fence.
    class MemFenceCpuSerial : public concepts::Implements<ConceptMemFence, MemFenceCpuSerial>
    {
    };

    namespace trait
    {
        template<>
        struct MemFence<MemFenceCpuSerial, memory_scope::Block>
        {
            static auto mem_fence(MemFenceCpuSerial const&, memory_scope::Block const&)
            {
                /* Nothing to be done on the block level for the serial case. */
            }
        };

        template<>
        struct MemFence<MemFenceCpuSerial, memory_scope::Grid>
        {
            static auto mem_fence(MemFenceCpuSerial const&, memory_scope::Grid const&)
            {
                /* Nothing to be done on the grid level for the serial case. */
            }
        };

        template<typename TMemScope>
        struct MemFence<MemFenceCpuSerial, TMemScope>
        {
            static auto mem_fence(MemFenceCpuSerial const&, TMemScope const&)
            {
                /* Enable device fences because we may want to synchronize with other (serial) kernels. */
                std::atomic_thread_fence(std::memory_order_acq_rel);
            }
        };
    } // namespace trait
} // namespace alpaka
