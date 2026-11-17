/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/grid/Traits.hpp"

namespace alpaka
{
    //! The NoOp grid synchronization for accelerators that only support a single thread with cooperative kernels.
    class GridSyncNoOp : public interface::Implements<ConceptGridSync, GridSyncNoOp>
    {
    };

    namespace trait
    {
        template<>
        struct SyncGridThreads<GridSyncNoOp>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncGridThreads(GridSyncNoOp const& /*gridSync*/) -> void
            {
                // Nothing to do.
            }
        };

    } // namespace trait

} // namespace alpaka
