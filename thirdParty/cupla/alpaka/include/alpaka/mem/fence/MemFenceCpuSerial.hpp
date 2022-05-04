/* Copyright 2022 Jan Stephan, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Concepts.hpp>
#include <alpaka/mem/fence/Traits.hpp>

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

                static std::atomic<int> dummy{42};

                /* ISO C++ fences are only clearly defined if there are atomic operations surrounding them. So we use
                 * these dummy operations to ensure this.*/
                auto x = dummy.load(std::memory_order_relaxed);
                std::atomic_thread_fence(std::memory_order_acq_rel);
                dummy.store(x, std::memory_order_relaxed);
            }
        };
    } // namespace trait
} // namespace alpaka
