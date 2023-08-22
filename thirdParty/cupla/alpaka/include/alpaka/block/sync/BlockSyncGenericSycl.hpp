/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/block/sync/Traits.hpp"

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL block synchronization.
    template<typename TDim>
    class BlockSyncGenericSycl : public concepts::Implements<ConceptBlockSync, BlockSyncGenericSycl<TDim>>
    {
    public:
        using BlockSyncBase = BlockSyncGenericSycl<TDim>;

        BlockSyncGenericSycl(sycl::nd_item<TDim::value> work_item) : my_item{work_item}
        {
        }

        sycl::nd_item<TDim::value> my_item;
    };
} // namespace alpaka

namespace alpaka::trait
{
    template<typename TDim>
    struct SyncBlockThreads<BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreads(BlockSyncGenericSycl<TDim> const& blockSync) -> void
        {
            blockSync.my_item.barrier();
        }
    };

    template<typename TDim>
    struct SyncBlockThreadsPredicate<BlockCount, BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const& blockSync, int predicate) -> int
        {
            auto const group = blockSync.my_item.get_group();
            blockSync.my_item.barrier();

            auto const counter = (predicate != 0) ? 1 : 0;
            return sycl::reduce_over_group(group, counter, sycl::plus<>{});
        }
    };

    template<typename TDim>
    struct SyncBlockThreadsPredicate<BlockAnd, BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const& blockSync, int predicate) -> int
        {
            auto const group = blockSync.my_item.get_group();
            blockSync.my_item.barrier();

            return static_cast<int>(sycl::all_of_group(group, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct SyncBlockThreadsPredicate<BlockOr, BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreadsPredicate(BlockSyncGenericSycl<TDim> const& blockSync, int predicate) -> int
        {
            auto const group = blockSync.my_item.get_group();
            blockSync.my_item.barrier();

            return static_cast<int>(sycl::any_of_group(group, static_cast<bool>(predicate)));
        }
    };
} // namespace alpaka::trait

#endif
