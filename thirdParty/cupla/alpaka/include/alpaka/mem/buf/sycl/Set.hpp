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
#    include <alpaka/dev/DevGenericSycl.hpp>
#    include <alpaka/dev/Traits.hpp>
#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/extent/Traits.hpp>
#    include <alpaka/mem/buf/sycl/Common.hpp>
#    include <alpaka/mem/view/Traits.hpp>
#    include <alpaka/queue/Traits.hpp>

#    include <CL/sycl.hpp>

#    include <cstddef>
#    include <cstdint>
#    include <memory>

namespace alpaka
{
    namespace experimental
    {
        namespace detail
        {
            template<std::size_t TDim>
            using Accessor = sycl::accessor<
                std::byte,
                TDim,
                sycl::access_mode::write,
                sycl::target::global_buffer,
                sycl::access::placeholder::true_t>;

            //! The SYCL memory set trait.
            template<typename TAccessor>
            struct TaskSetSycl
            {
                auto operator()(sycl::handler& cgh) const -> void
                {
                    cgh.require(m_accessor);
                    cgh.fill(m_accessor, m_value);
                }

                TAccessor m_accessor;
                std::byte m_value;
                // Distinguish from non-alpaka types (= host tasks)
                static constexpr auto is_sycl_task = true;
            };
        } // namespace detail
    } // namespace experimental

    namespace trait
    {
        //! The SYCL device memory set trait specialization.
        template<typename TDim, typename TPltf>
        struct CreateTaskMemset<TDim, experimental::DevGenericSycl<TPltf>>
        {
            template<typename TExtent, typename TViewFwd>
            static auto createTaskMemset(TViewFwd&& view, std::uint8_t const& byte, TExtent const& ext)
            {
                ALPAKA_DEBUG_FULL_LOG_SCOPE;

                constexpr auto set_dim = static_cast<int>(Dim<TExtent>::value);
                using TView = std::remove_reference_t<TViewFwd>;
                using ElemType = Elem<TView>;
                using DstType = alpaka::experimental::detail::Accessor<set_dim>;

                // Reinterpret as byte buffer
                auto buf = view.m_buffer.template reinterpret<std::byte>();
                auto const byte_val = static_cast<std::byte>(byte);

                auto const range = experimental::detail::make_sycl_range(ext, sizeof(ElemType));
                return experimental::detail::TaskSetSycl<DstType>{DstType{buf, range}, byte_val};
            }
        };
    } // namespace trait
} // namespace alpaka

#endif
