/* Copyright 2022 Jan Stephan, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Concepts.hpp"

namespace alpaka
{
    struct ConceptMemFence
    {
    };

    namespace memory_scope
    {
        //! Memory fences are observed by all threads in the same block.
        struct Block
        {
        };

        //! Memory fences are observed by all threads in the same grid.
        struct Grid
        {
        };

        //! Memory fences are observed by all threads on the device.
        struct Device
        {
        };
    } // namespace memory_scope

    //! The memory fence trait.
    namespace trait
    {
        //! The mem_fence trait.
        template<typename TMemFence, typename TMemScope, typename TSfinae = void>
        struct MemFence;
    } // namespace trait

    //! Issues memory fence instructions.
    //
    // Issues a memory fence instruction for a given memory scope (\a memory_scope::Block or \a memory_scope::Device).
    // This guarantees the following:
    // * All \a LOAD instructions preceeding the fence will always occur before the LOAD instructions following the
    //   fence (\a LoadLoad coherence)
    // * All \a STORE instructions preceeding the fence will always occur before the STORE instructions following the
    //   fence (\a LoadStore coherence). The pre-fence STORE results will be propagated to the other threads in the
    //   scope at an unknown point in time.
    //
    // Note that there are no further guarantees, especially with regard to \a LoadStore ordering. Users should not
    // mistake this as a synchronization function between threads (please use syncBlockThreads() instead).
    //
    //! \tparam TMemFence The memory fence implementation type.
    //! \tparam TMemScope The memory scope type.
    //! \param fence The memory fence implementation.
    //! \param scope The memory scope.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TMemFence, typename TMemScope>
    ALPAKA_FN_ACC auto mem_fence(TMemFence const& fence, TMemScope const& scope) -> void
    {
        using ImplementationBase = concepts::ImplementationBase<ConceptMemFence, TMemFence>;
        trait::MemFence<ImplementationBase, TMemScope>::mem_fence(fence, scope);
    }
} // namespace alpaka
