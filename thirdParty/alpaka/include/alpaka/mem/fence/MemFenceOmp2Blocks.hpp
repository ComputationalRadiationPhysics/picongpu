/* Copyright 2022 Jan Stephan, Bernhard Manfred Gruber, Andrea Bocci, Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Interface.hpp"
#include "alpaka/mem/fence/MemFenceOmp2Order.hpp"
#include "alpaka/mem/fence/Traits.hpp"

#ifdef ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED

#    if _OPENMP < 200203
#        error If ALPAKA_ACC_CPU_B_OMP2_T_SEQ_ENABLED is set, the compiler has to support OpenMP 2.0 or higher!
#    endif

namespace alpaka
{
    //! The CPU OpenMP 2.0 block memory fence.
    class MemFenceOmp2Blocks : public interface::Implements<ConceptMemFence, MemFenceOmp2Blocks>
    {
    };

    namespace trait
    {
        template<>
        struct MemFenceDefaultOrder<MemFenceOmp2Blocks>
        {
            using type = mem_order::AcqRel;
            static constexpr auto value = mem_order::acq_rel;
        };

        template<MemoryOrder TMemOrder>
        struct MemFence<MemFenceOmp2Blocks, TMemOrder, memory_scope::Block>
        {
            static auto mem_fence(MemFenceOmp2Blocks const&, TMemOrder, memory_scope::Block const&)
            {
                // Only one thread per block allowed -> no memory fence required on block level
            }
        };

        template<MemoryOrder TMemOrder>
        struct MemFence<MemFenceOmp2Blocks, TMemOrder, memory_scope::Grid>
        {
            static auto mem_fence(MemFenceOmp2Blocks const&, TMemOrder order, memory_scope::Grid const&)
            {
                alpaka::detail::flushOmp(order);
            }
        };

        template<MemoryOrder TMemOrder>
        struct MemFence<MemFenceOmp2Blocks, TMemOrder, memory_scope::Device>
        {
            static auto mem_fence(MemFenceOmp2Blocks const&, TMemOrder order, memory_scope::Device const&)
            {
                alpaka::detail::flushOmp(order);
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
