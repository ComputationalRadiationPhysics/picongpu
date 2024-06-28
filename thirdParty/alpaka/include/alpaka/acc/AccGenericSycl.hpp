/* Copyright 2024 Jan Stephan, Antonio Di Pilato, Andrea Bocci, Luca Ferragina, Aurora Perego
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Base classes.
#include "alpaka/atomic/AtomicGenericSycl.hpp"
#include "alpaka/atomic/AtomicHierarchy.hpp"
#include "alpaka/block/shared/dyn/BlockSharedMemDynGenericSycl.hpp"
#include "alpaka/block/shared/st/BlockSharedMemStGenericSycl.hpp"
#include "alpaka/block/sync/BlockSyncGenericSycl.hpp"
#include "alpaka/idx/bt/IdxBtGenericSycl.hpp"
#include "alpaka/idx/gb/IdxGbGenericSycl.hpp"
#include "alpaka/intrinsic/IntrinsicGenericSycl.hpp"
#include "alpaka/math/MathGenericSycl.hpp"
#include "alpaka/mem/fence/MemFenceGenericSycl.hpp"
#include "alpaka/rand/RandDefault.hpp"
#include "alpaka/rand/RandGenericSycl.hpp"
#include "alpaka/warp/WarpGenericSycl.hpp"
#include "alpaka/workdiv/WorkDivGenericSycl.hpp"

// Specialized traits.
#include "alpaka/acc/Traits.hpp"
#include "alpaka/dev/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/kernel/Traits.hpp"
#include "alpaka/platform/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

// Implementation details.
#include "alpaka/core/BoostPredef.hpp"
#include "alpaka/core/ClipCast.hpp"
#include "alpaka/core/Sycl.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <sycl/sycl.hpp>

namespace alpaka
{
    //! The SYCL accelerator.
    //!
    //! This accelerator allows parallel kernel execution on SYCL devices.
    template<typename TDim, typename TIdx>
    class AccGenericSycl
        : public WorkDivGenericSycl<TDim, TIdx>
        , public gb::IdxGbGenericSycl<TDim, TIdx>
        , public bt::IdxBtGenericSycl<TDim, TIdx>
        , public AtomicHierarchy<AtomicGenericSycl, AtomicGenericSycl, AtomicGenericSycl>
        , public math::MathGenericSycl
        , public BlockSharedMemDynGenericSycl
        , public BlockSharedMemStGenericSycl
        , public BlockSyncGenericSycl<TDim>
        , public IntrinsicGenericSycl
        , public MemFenceGenericSycl
#    ifdef ALPAKA_DISABLE_VENDOR_RNG
        , public rand::RandDefault
#    else
        , public rand::RandGenericSycl<TDim>
#    endif
        , public warp::WarpGenericSycl<TDim>
    {
        static_assert(TDim::value > 0, "The SYCL accelerator must have a dimension greater than zero.");

    public:
        AccGenericSycl(AccGenericSycl const&) = delete;
        AccGenericSycl(AccGenericSycl&&) = delete;
        auto operator=(AccGenericSycl const&) -> AccGenericSycl& = delete;
        auto operator=(AccGenericSycl&&) -> AccGenericSycl& = delete;

        AccGenericSycl(
            Vec<TDim, TIdx> const& threadElemExtent,
            sycl::nd_item<TDim::value> work_item,
            sycl::local_accessor<std::byte> dyn_shared_acc,
            sycl::local_accessor<std::byte> st_shared_acc)
            : WorkDivGenericSycl<TDim, TIdx>{threadElemExtent, work_item}
            , gb::IdxGbGenericSycl<TDim, TIdx>{work_item}
            , bt::IdxBtGenericSycl<TDim, TIdx>{work_item}
            , BlockSharedMemDynGenericSycl{dyn_shared_acc}
            , BlockSharedMemStGenericSycl{st_shared_acc}
            , BlockSyncGenericSycl<TDim>{work_item}
#    ifndef ALPAKA_DISABLE_VENDOR_RNG
            , rand::RandGenericSycl<TDim>{work_item}
#    endif
            , warp::WarpGenericSycl<TDim>{work_item}
        {
        }
    };
} // namespace alpaka

namespace alpaka::trait
{
    //! The SYCL accelerator type trait specialization.
    template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
    struct AccType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
    {
        using type = TAcc<TDim, TIdx>;
    };

    //! The SYCL single thread accelerator type trait specialization.
    template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
    struct IsSingleThreadAcc<
        TAcc<TDim, TIdx>,
        std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>> : std::false_type
    {
    };

    //! The SYCL multi thread accelerator type trait specialization.
    template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
    struct IsMultiThreadAcc<
        TAcc<TDim, TIdx>,
        std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>> : std::true_type
    {
    };

    //! The SYCL accelerator device properties get trait specialization.
    template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
    struct GetAccDevProps<
        TAcc<TDim, TIdx>,
        std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
    {
        static auto getAccDevProps(typename DevType<TAcc<TDim, TIdx>>::type const& dev) -> AccDevProps<TDim, TIdx>
        {
            auto const device = dev.getNativeHandle().first;
            auto const max_threads_dim
                = device.template get_info<sycl::info::device::max_work_item_sizes<TDim::value>>();
            Vec<TDim, TIdx> max_threads_dim_vec{};
            for(int i = 0; i < static_cast<int>(TDim::value); i++)
                max_threads_dim_vec[i] = alpaka::core::clipCast<TIdx>(max_threads_dim[i]);
            return {// m_multiProcessorCount
                    alpaka::core::clipCast<TIdx>(device.template get_info<sycl::info::device::max_compute_units>()),
                    // m_gridBlockExtentMax
                    getExtentVecEnd<TDim>(Vec<DimInt<3u>, TIdx>(
                        // WARNING: There is no SYCL way to determine these values
                        std::numeric_limits<TIdx>::max(),
                        std::numeric_limits<TIdx>::max(),
                        std::numeric_limits<TIdx>::max())),
                    // m_gridBlockCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_blockThreadExtentMax
                    max_threads_dim_vec,
                    // m_blockThreadCountMax
                    alpaka::core::clipCast<TIdx>(device.template get_info<sycl::info::device::max_work_group_size>()),
                    // m_threadElemExtentMax
                    Vec<TDim, TIdx>::all(std::numeric_limits<TIdx>::max()),
                    // m_threadElemCountMax
                    std::numeric_limits<TIdx>::max(),
                    // m_sharedMemSizeBytes
                    device.template get_info<sycl::info::device::local_mem_size>(),
                    // m_globalMemSizeBytes
                    getMemBytes(dev)};
        }
    };

    //! The SYCL accelerator dimension getter trait specialization.
    template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
    struct DimType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
    {
        using type = TDim;
    };

    //! The SYCL accelerator idx type trait specialization.
    template<template<typename, typename> typename TAcc, typename TDim, typename TIdx>
    struct IdxType<TAcc<TDim, TIdx>, std::enable_if_t<std::is_base_of_v<AccGenericSycl<TDim, TIdx>, TAcc<TDim, TIdx>>>>
    {
        using type = TIdx;
    };
} // namespace alpaka::trait

#endif
