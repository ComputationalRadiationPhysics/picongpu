/* Copyright 2022 Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Concepts.hpp"
#include "alpaka/mem/fence/Traits.hpp"

#ifdef ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_SEQ_T_OMP2_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

namespace alpaka
{
    //! The CPU OpenMP 2.0 block memory fence.
    class MemFenceOmp2Threads : public concepts::Implements<ConceptMemFence, MemFenceOmp2Threads>
    {
    };

    namespace trait
    {
        template<typename TMemScope>
        struct MemFence<MemFenceOmp2Threads, TMemScope>
        {
            static auto mem_fence(MemFenceOmp2Threads const&, TMemScope const&)
            {
                /*
                 * Intuitively, this pragma creates a fence on the block level.
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
#    pragma omp flush
#    ifdef _MSC_VER
                ; // MSVC needs an empty statement here or it diagnoses a syntax error
#    endif
            }
        };
    } // namespace trait
} // namespace alpaka
#endif
