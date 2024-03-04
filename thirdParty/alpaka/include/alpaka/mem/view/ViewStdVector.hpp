/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

/* TODO: Once C++20 is available remove this file and replace with a generic ContiguousContainer solution based on
 * concepts. It should be sufficient to check for the existence of Container.size() and Container.data() */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/dev/DevCpu.hpp"
#include "alpaka/mem/view/Traits.hpp"
#include "alpaka/platform/PlatformCpu.hpp"

#include <vector>

namespace alpaka::trait
{
    //! The std::vector device type trait specialization.
    template<typename TElem, typename TAllocator>
    struct DevType<std::vector<TElem, TAllocator>>
    {
        using type = DevCpu;
    };

    //! The std::vector device get trait specialization.
    template<typename TElem, typename TAllocator>
    struct GetDev<std::vector<TElem, TAllocator>>
    {
        ALPAKA_FN_HOST static auto getDev(std::vector<TElem, TAllocator> const& /* view */) -> DevCpu
        {
            return getDevByIdx(PlatformCpu{}, 0u);
        }
    };

    //! The std::vector dimension getter trait specialization.
    template<typename TElem, typename TAllocator>
    struct DimType<std::vector<TElem, TAllocator>>
    {
        using type = DimInt<1u>;
    };

    //! The std::vector memory element type get trait specialization.
    template<typename TElem, typename TAllocator>
    struct ElemType<std::vector<TElem, TAllocator>>
    {
        using type = TElem;
    };

    template<typename TElem, typename TAllocator>
    struct GetExtents<std::vector<TElem, TAllocator>>
    {
        ALPAKA_FN_HOST constexpr auto operator()(std::vector<TElem, TAllocator> const& a)
            -> Vec<DimInt<1>, Idx<std::vector<TElem, TAllocator>>>
        {
            return {std::size(a)};
        }
    };

    //! The std::vector native pointer get trait specialization.
    template<typename TElem, typename TAllocator>
    struct GetPtrNative<std::vector<TElem, TAllocator>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(std::vector<TElem, TAllocator> const& view) -> TElem const*
        {
            return std::data(view);
        }

        ALPAKA_FN_HOST static auto getPtrNative(std::vector<TElem, TAllocator>& view) -> TElem*
        {
            return std::data(view);
        }
    };

    //! The std::vector offset get trait specialization.
    template<typename TElem, typename TAllocator>
    struct GetOffsets<std::vector<TElem, TAllocator>>
    {
        ALPAKA_FN_HOST auto operator()(std::vector<TElem, TAllocator> const&) const
            -> Vec<DimInt<1>, Idx<std::vector<TElem, TAllocator>>>
        {
            return {0};
        }
    };

    //! The std::vector idx type trait specialization.
    template<typename TElem, typename TAllocator>
    struct IdxType<std::vector<TElem, TAllocator>>
    {
        using type = std::size_t;
    };
} // namespace alpaka::trait
