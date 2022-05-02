/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/* TODO: Once C++20 is available remove this file and replace with a generic ContiguousContainer solution based on
 * concepts. It should be sufficient to check for the existence of Container.size() and Container.data() */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/pltf/PltfCpu.hpp>

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
            return getDevByIdx<PltfCpu>(0u);
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

    //! The std::array width get trait specialization.
    template<typename TElem, std::size_t Tsize>
    struct GetExtent<DimInt<0u>, std::array<TElem, Tsize>>
    {
        ALPAKA_FN_HOST static constexpr auto getExtent(std::array<TElem, Tsize> const& extent)
            -> Idx<std::array<TElem, Tsize>>
        {
            return std::size(extent);
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
    template<typename TIdx, typename TElem, std::size_t Tsize>
    struct GetOffset<TIdx, std::array<TElem, Tsize>>
    {
        ALPAKA_FN_HOST static auto getOffset(std::array<TElem, Tsize> const&) -> Idx<std::array<TElem, Tsize>>
        {
            return 0u;
        }
    };

    //! The std::vector idx type trait specialization.
    template<typename TElem, std::size_t Tsize>
    struct IdxType<std::array<TElem, Tsize>>
    {
        using type = std::size_t;
    };
} // namespace alpaka::trait
