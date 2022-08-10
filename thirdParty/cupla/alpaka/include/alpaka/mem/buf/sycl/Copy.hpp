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

#    include <alpaka/core/Debug.hpp>
#    include <alpaka/core/Sycl.hpp>
#    include <alpaka/dev/DevCpu.hpp>
#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/elem/Traits.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/mem/buf/sycl/Common.hpp>
#    include <alpaka/mem/view/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <memory>
#    include <type_traits>

namespace alpaka::experimental::detail
{
    template<typename TElem, std::size_t TDim>
    using SrcAccessor = sycl::
        accessor<TElem, TDim, sycl::access_mode::read, sycl::target::global_buffer, sycl::access::placeholder::true_t>;

    template<typename TElem, std::size_t TDim>
    using DstAccessor = sycl::accessor<
        TElem,
        TDim,
        sycl::access_mode::write,
        sycl::target::global_buffer,
        sycl::access::placeholder::true_t>;

    enum class Direction
    {
        h2d,
        d2h,
        d2d
    };

    template<typename TSrc, typename TDst, Direction TDirection>
    struct TaskCopySycl
    {
        auto operator()(sycl::handler& cgh) const -> void
        {
            if constexpr(TDirection == Direction::d2h || TDirection == Direction::d2d)
                cgh.require(m_src);

            if constexpr(TDirection == Direction::h2d || TDirection == Direction::d2d)
                cgh.require(m_dst);

            cgh.copy(m_src, m_dst);
        }

        TSrc m_src;
        TDst m_dst;
        static constexpr auto is_sycl_task = true;
    };
} // namespace alpaka::experimental::detail

// Trait specializations for CreateTaskMemcpy.
namespace alpaka::trait
{
    //! The SYCL host-to-device memory copy trait specialization.
    template<typename TDim, typename TPltf>
    struct CreateTaskMemcpy<TDim, experimental::DevGenericSycl<TPltf>, DevCpu>
    {
        template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
        static auto createTaskMemcpy(TViewDstFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& ext)
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            constexpr auto copy_dim = static_cast<int>(Dim<TExtent>::value);
            using ElemType = Elem<std::remove_const_t<TViewSrc>>;
            using SrcType = ElemType const*;
            using DstType = alpaka::experimental::detail::DstAccessor<ElemType, copy_dim>;

            auto const range = experimental::detail::make_sycl_range(ext);
            auto const offset = experimental::detail::make_sycl_offset(viewDst);

            return experimental::detail::TaskCopySycl<SrcType, DstType, experimental::detail::Direction::h2d>{
                getPtrNative(viewSrc),
                DstType{viewDst.m_buffer, range, offset}};
        }
    };

    //! The SYCL device-to-host memory copy trait specialization.
    template<typename TDim, typename TPltf>
    struct CreateTaskMemcpy<TDim, DevCpu, experimental::DevGenericSycl<TPltf>>
    {
        template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
        static auto createTaskMemcpy(TViewDstFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& ext)
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            constexpr auto copy_dim = static_cast<int>(Dim<TExtent>::value);
            using ElemType = Elem<std::remove_const_t<TViewSrc>>;
            using SrcType = alpaka::experimental::detail::SrcAccessor<ElemType, copy_dim>;
            using DstType = ElemType*;

            auto const range = experimental::detail::make_sycl_range(ext);
            auto const offset = experimental::detail::make_sycl_offset(viewSrc);

            auto view_src = const_cast<TViewSrc&>(viewSrc);

            return experimental::detail::TaskCopySycl<SrcType, DstType, experimental::detail::Direction::d2h>{
                SrcType{view_src.m_buffer, range, offset},
                getPtrNative(viewDst)};
        }
    };

    //! The SYCL device-to-device memory copy trait specialization.
    template<typename TDim, typename TPltfDst, typename TPltfSrc>
    struct CreateTaskMemcpy<TDim, experimental::DevGenericSycl<TPltfDst>, experimental::DevGenericSycl<TPltfSrc>>
    {
        template<typename TExtent, typename TViewSrc, typename TViewDstFwd>
        static auto createTaskMemcpy(TViewDstFwd&& viewDst, TViewSrc const& viewSrc, TExtent const& ext)
        {
            ALPAKA_DEBUG_FULL_LOG_SCOPE;

            constexpr auto copy_dim = static_cast<int>(Dim<TExtent>::value);
            using ElemType = Elem<std::remove_const_t<TViewSrc>>;
            using SrcType = alpaka::experimental::detail::SrcAccessor<ElemType, copy_dim>;
            using DstType = alpaka::experimental::detail::DstAccessor<ElemType, copy_dim>;

            auto const range = experimental::detail::make_sycl_range(ext);
            auto const offset_src = experimental::detail::make_sycl_offset(viewSrc);
            auto const offset_dst = experimental::detail::make_sycl_offset(viewDst);

            auto view_src = const_cast<TViewSrc&>(viewSrc);

            return experimental::detail::TaskCopySycl<SrcType, DstType, experimental::detail::Direction::d2d>{
                SrcType{view_src.m_buffer, range, offset_src},
                DstType{viewDst.m_buffer, range, offset_dst}};
        }
    };
} // namespace alpaka::trait

#endif
