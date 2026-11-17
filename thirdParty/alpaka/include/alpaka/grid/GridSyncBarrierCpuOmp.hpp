/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/grid/Traits.hpp"

#ifdef _OPENMP

namespace alpaka
{
    //! The grid synchronization for OMP accelerators.
    class GridSyncOmp : public interface::Implements<ConceptGridSync, GridSyncOmp>
    {
    };

    namespace trait
    {
        template<>
        struct SyncGridThreads<GridSyncOmp>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncGridThreads(GridSyncOmp const& /*gridSync*/) -> void
            {
#    pragma omp barrier
            }
        };

    } // namespace trait

} // namespace alpaka

#endif
