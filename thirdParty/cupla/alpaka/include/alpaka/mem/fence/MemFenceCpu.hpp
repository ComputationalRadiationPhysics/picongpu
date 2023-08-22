/* Copyright 2022 Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/mem/fence/Traits.hpp"

#include <atomic>

namespace alpaka
{
    //! The default CPU memory fence.
    class MemFenceCpu : public concepts::Implements<ConceptMemFence, MemFenceCpu>
    {
    };

    namespace trait
    {
        template<typename TMemScope>
        struct MemFence<MemFenceCpu, TMemScope>
        {
            static auto mem_fence(MemFenceCpu const&, TMemScope const&)
            {
                /*
                 * Intuitively, std::atomic_thread_fence creates a fence on the block level.
                 *
                 * Creating a block fence is enough for the whole device because the blocks are executed serially. By
                 * definition of fences, preceding blocks don't have a guarantee to see the results of this block's
                 * STORE operations (only that they will be ordered correctly); the following blocks see the results
                 * once they start. Consider the following code:
                 *
                 * int x = 1;
                 * int y = 2;
                 *
                 * void foo()
                 * {
                 *     x = 10;
                 *     alpaka::mem_fence(acc, memory_scope::device);
                 *     y = 20;
                 * }
                 *
                 * void bar()
                 * {
                 *     auto b = y;
                 *     alpaka::mem_fence(acc, memory_scope::device);
                 *     auto a = x;
                 * }
                 *
                 * The following are all valid outcomes:
                 *   a == 1 && b == 2
                 *   a == 10 && b == 2
                 *   a == 10 && b == 20
                 */

                std::atomic_thread_fence(std::memory_order_acq_rel);
            }
        };
    } // namespace trait
} // namespace alpaka
