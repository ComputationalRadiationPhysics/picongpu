/* Copyright 2026 Mykhailo Varvarin
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/core/Interface.hpp"
#include "alpaka/grid/Traits.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/ext/oneapi/experimental/root_group.hpp>
#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The grid synchronization for SYCL.
    template<typename TDim>
    class GridSyncGenericSycl : public interface::Implements<ConceptGridSync, GridSyncGenericSycl<TDim>>
    {
    public:
        GridSyncGenericSycl(sycl::nd_item<TDim::value> work_item) : my_item{work_item}
        {
        }

        sycl::nd_item<TDim::value> my_item;
    };

    namespace trait
    {
        template<typename TDim>
        struct SyncGridThreads<GridSyncGenericSycl<TDim>>
        {
            ALPAKA_NO_HOST_ACC_WARNING
            ALPAKA_FN_ACC static auto syncGridThreads(GridSyncGenericSycl<TDim> const& gridSync) -> void
            {
                sycl::group_barrier(gridSync.my_item.ext_oneapi_get_root_group());
            }
        };

    } // namespace trait

} // namespace alpaka

#endif
