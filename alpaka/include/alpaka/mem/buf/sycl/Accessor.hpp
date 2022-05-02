/* Copyright 2021 Jan Stephan
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#ifdef ALPAKA_ACC_SYCL_ENABLED

#    include <alpaka/dim/DimIntegralConst.hpp>
#    include <alpaka/mem/view/Accessor.hpp>
#    include <alpaka/mem/view/ViewAccessor.hpp>

#    include <CL/sycl.hpp>

#    include <cstddef>
#    include <utility>

namespace alpaka::experimental
{
    template<typename TElem, typename TDim, typename TIdx, typename TDev>
    class BufGenericSycl;

    namespace detail
    {
        template<typename... TAlpakaAccessModes>
        inline constexpr auto sycl_access_mode = sycl::access_mode::read_write;

        template<>
        inline constexpr auto sycl_access_mode<ReadAccess> = sycl::access_mode::read;

        template<>
        inline constexpr auto sycl_access_mode<WriteAccess> = sycl::access_mode::write;

        template<typename TElem, int TDim, typename... TAlpakaAccessModes>
        using SyclAccessor = sycl::accessor<
            TElem,
            TDim,
            sycl_access_mode<TAlpakaAccessModes...>,
            sycl::target::global_buffer,
            sycl::access::placeholder::true_t>;
    } // namespace detail

    template<typename TElem, typename TIdx, typename TAccessModes>
    struct Accessor<detail::SyclAccessor<TElem, 1, TAccessModes>, TElem, TIdx, std::size_t{1}, TAccessModes>
    {
        static constexpr auto sycl_access_mode = detail::sycl_access_mode<TAccessModes>;
        using SyclAccessor = detail::SyclAccessor<TElem, 1, TAccessModes>;
        using VecType = Vec<DimInt<1>, TIdx>;
        using ReturnType = std::conditional_t<
            std::is_same_v<TAccessModes, ReadAccess>,
            typename SyclAccessor::const_reference,
            typename SyclAccessor::reference>;

        Accessor(SyclAccessor accessor, Vec<DimInt<1>, TIdx> ext) : m_accessor{accessor}, extents{ext}
        {
        }

        auto operator[](VecType const& i) const -> ReturnType
        {
            auto const range = sycl::id<1>{i[0]};
            return m_accessor[range];
        }

        auto operator[](TIdx i) const -> ReturnType
        {
            return m_accessor[i];
        }

        template<typename... TIs>
        auto operator()(TIs... is) const
        {
            static_assert(sizeof...(TIs) == 1, "Number of indices must match the dimensionality.");
            return operator[](VecType{static_cast<TIdx>(is)...});
        }

        SyclAccessor m_accessor;
        VecType extents;
    };

    template<typename TElem, typename TIdx, std::size_t TDim, typename TAccessModes>
    struct Accessor<detail::SyclAccessor<TElem, DimInt<TDim>::value, TAccessModes>, TElem, TIdx, TDim, TAccessModes>
    {
        static constexpr auto sycl_access_mode = detail::sycl_access_mode<TAccessModes>;
        using SyclAccessor = detail::SyclAccessor<TElem, DimInt<TDim>::value, TAccessModes>;
        using VecType = Vec<DimInt<TDim>, TIdx>;
        using ReturnType = std::conditional_t<
            std::is_same_v<TAccessModes, ReadAccess>,
            typename SyclAccessor::const_reference,
            typename SyclAccessor::reference>;


        Accessor(SyclAccessor accessor, Vec<DimInt<TDim>, TIdx> ext) : m_accessor{accessor}, extents{ext}
        {
        }

        auto operator[](VecType const& i) const -> ReturnType
        {
            using IdType = sycl::id<DimInt<TDim>::value>;
            auto const id = (TDim == 2) ? IdType{i[1], i[0]} : IdType{i[2], i[1], i[0]};
            return m_accessor[id];
        }

        template<typename... TIs>
        auto operator()(TIs... is) const -> ReturnType
        {
            static_assert(sizeof...(TIs) == TDim, "Number of indices must match the dimensionality.");
            return operator[](VecType{static_cast<TIdx>(is)...});
        }

        SyclAccessor m_accessor;
        VecType extents;
    };

    namespace trait
    {
        namespace internal
        {
            template<typename TElem, typename TDim, typename TIdx, typename TDev>
            struct IsView<BufGenericSycl<TElem, TDim, TIdx, TDev>> : std::false_type
            {
            };
        } // namespace internal

        template<typename TElem, typename TDim, typename TIdx, typename TDev>
        struct BuildAccessor<BufGenericSycl<TElem, TDim, TIdx, TDev>>
        {
            template<typename... TAccessModes>
            static auto buildAccessor(BufGenericSycl<TElem, TDim, TIdx, TDev>& buffer)
            {
                using SyclAccessor = experimental::detail::SyclAccessor<TElem, TDim::value, TAccessModes...>;
                return Accessor<SyclAccessor, TElem, TIdx, TDim::value, TAccessModes...>{
                    SyclAccessor{buffer.m_buffer},
                    buffer.m_extentElements};
            }
        };
    } // namespace trait
} // namespace alpaka::experimental

#endif
