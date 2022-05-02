/* Copyright 2022 Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

#    include <alpaka/core/Concepts.hpp>
#    include <alpaka/mem/fence/Traits.hpp>

namespace alpaka
{
    //! The CPU OpenMP 2.0 block memory fence.
    class MemFenceOmp2Blocks : public concepts::Implements<ConceptMemFence, MemFenceOmp2Blocks>
    {
    };

    namespace trait
    {
        template<>
        struct MemFence<MemFenceOmp2Blocks, memory_scope::Block>
        {
            static auto mem_fence(MemFenceOmp2Blocks const&, memory_scope::Block const&)
            {
                // Only one thread per block allowed -> no memory fence required on block level
            }
        };

        template<>
        struct MemFence<MemFenceOmp2Blocks, memory_scope::Grid>
        {
            static auto mem_fence(MemFenceOmp2Blocks const&, memory_scope::Grid const&)
            {
#    pragma omp flush
            }
        };

        template<>
        struct MemFence<MemFenceOmp2Blocks, memory_scope::Device>
        {
            static auto mem_fence(MemFenceOmp2Blocks const&, memory_scope::Device const&)
            {
#    pragma omp flush
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
