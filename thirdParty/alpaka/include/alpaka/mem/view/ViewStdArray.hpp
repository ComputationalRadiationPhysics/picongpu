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

#include <array>

namespace alpaka::trait
{
    //! The std::array device type trait specialization.
    template<typename TElem, std::size_t Tsize>
    struct DevType<std::array<TElem, Tsize>>
    {
        using type = DevCpu;
    };

    //! The std::array device get trait specialization.
    template<typename TElem, std::size_t Tsize>
    struct GetDev<std::array<TElem, Tsize>>
    {
        ALPAKA_FN_HOST static auto getDev(std::array<TElem, Tsize> const& /* view */) -> DevCpu
        {
            // Instantiating the CPU platform here is a hack we can do internally, because we know that the CPU
            // platform does not contain any data. But it generally does not apply.
            return getDevByIdx(PlatformCpu{}, 0u);
        }
    };

    //! The std::array dimension getter trait specialization.
    template<typename TElem, std::size_t Tsize>
    struct DimType<std::array<TElem, Tsize>>
    {
        using type = DimInt<1u>;
    };

    //! The std::array memory element type get trait specialization.
    template<typename TElem, std::size_t Tsize>
    struct ElemType<std::array<TElem, Tsize>>
    {
        using type = TElem;
    };

    template<typename TElem, std::size_t Tsize>
    struct GetExtents<std::array<TElem, Tsize>>
    {
        ALPAKA_FN_HOST constexpr auto operator()(std::array<TElem, Tsize> const& a)
            -> Vec<DimInt<1>, Idx<std::array<TElem, Tsize>>>
        {
            return {std::size(a)};
        }
    };

    //! The std::array native pointer get trait specialization.
    template<typename TElem, std::size_t Tsize>
    struct GetPtrNative<std::array<TElem, Tsize>>
    {
        ALPAKA_FN_HOST static auto getPtrNative(std::array<TElem, Tsize> const& view) -> TElem const*
        {
            return std::data(view);
        }

        ALPAKA_FN_HOST static auto getPtrNative(std::array<TElem, Tsize>& view) -> TElem*
        {
            return std::data(view);
        }
    };

    //! The std::array offset get trait specialization.
    template<typename TElem, std::size_t Tsize>
    struct GetOffsets<std::array<TElem, Tsize>>
    {
        ALPAKA_FN_HOST auto operator()(std::array<TElem, Tsize> const&)
            -> Vec<DimInt<1>, Idx<std::array<TElem, Tsize>>>
        {
            return {0};
        }
    };

    //! The std::vector idx type trait specialization.
    template<typename TElem, std::size_t Tsize>
    struct IdxType<std::array<TElem, Tsize>>
    {
        using type = std::size_t;
    };
} // namespace alpaka::trait
