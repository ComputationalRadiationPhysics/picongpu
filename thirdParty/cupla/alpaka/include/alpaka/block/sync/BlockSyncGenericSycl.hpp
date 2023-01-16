/* Copyright 2022 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/block/sync/Traits.hpp>

#    include <CL/sycl.hpp>

namespace alpaka::experimental
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
} // namespace alpaka::experimental

namespace alpaka::trait
{
    template<typename TDim>
    struct SyncBlockThreads<experimental::BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreads(experimental::BlockSyncGenericSycl<TDim> const& blockSync) -> void
        {
            blockSync.my_item.barrier();
        }
    };

    template<typename TDim>
    struct SyncBlockThreadsPredicate<BlockCount, experimental::BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreadsPredicate(experimental::BlockSyncGenericSycl<TDim> const& blockSync, int predicate)
            -> int
        {
            auto const group = blockSync.my_item.get_group();
            blockSync.my_item.barrier();

            auto const counter = (predicate != 0) ? 1 : 0;
            return sycl::reduce_over_group(group, counter, sycl::plus<>{});
        }
    };

    template<typename TDim>
    struct SyncBlockThreadsPredicate<BlockAnd, experimental::BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreadsPredicate(experimental::BlockSyncGenericSycl<TDim> const& blockSync, int predicate)
            -> int
        {
            auto const group = blockSync.my_item.get_group();
            blockSync.my_item.barrier();

            return static_cast<int>(sycl::all_of_group(group, static_cast<bool>(predicate)));
        }
    };

    template<typename TDim>
    struct SyncBlockThreadsPredicate<BlockOr, experimental::BlockSyncGenericSycl<TDim>>
    {
        static auto syncBlockThreadsPredicate(experimental::BlockSyncGenericSycl<TDim> const& blockSync, int predicate)
            -> int
        {
            auto const group = blockSync.my_item.get_group();
            blockSync.my_item.barrier();

            return static_cast<int>(sycl::any_of_group(group, static_cast<bool>(predicate)));
        }
    };
} // namespace alpaka::trait

#endif
