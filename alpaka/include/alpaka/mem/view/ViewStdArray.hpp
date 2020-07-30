/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dev/DevCpu.hpp>
#include <alpaka/mem/view/Traits.hpp>
#include <alpaka/pltf/PltfCpu.hpp>

#include <array>

namespace alpaka
{
    namespace dev
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array device type trait specialization.
            template<
                typename TElem,
                std::size_t Tsize>
            struct DevType<
                std::array<TElem, Tsize>>
            {
                using type = dev::DevCpu;
            };

            //#############################################################################
            //! The std::array device get trait specialization.
            template<
                typename TElem,
                std::size_t Tsize>
            struct GetDev<
                std::array<TElem, Tsize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getDev(
                    std::array<TElem, Tsize> const & view)
                -> dev::DevCpu
                {
                    alpaka::ignore_unused(view);
                    return pltf::getDevByIdx<pltf::PltfCpu>(0u);
                }
            };
        }
    }
    namespace dim
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array dimension getter trait specialization.
            template<
                typename TElem,
                std::size_t Tsize>
            struct DimType<
                std::array<TElem, Tsize>>
            {
                using type = dim::DimInt<1u>;
            };
        }
    }
    namespace elem
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array memory element type get trait specialization.
            template<
                typename TElem,
                std::size_t Tsize>
            struct ElemType<
                std::array<TElem, Tsize>>
            {
                using type = TElem;
            };
        }
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array width get trait specialization.
            template<
                typename TElem,
                std::size_t Tsize>
            struct GetExtent<
                dim::DimInt<0u>,
                std::array<TElem, Tsize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static constexpr auto getExtent(
                    std::array<TElem, Tsize> const & extent)
                -> idx::Idx<std::array<TElem, Tsize>>
                {
                    alpaka::ignore_unused(extent);
                    return Tsize;
                }
            };
        }
    }
    namespace mem
    {
        namespace view
        {
            namespace traits
            {
                //#############################################################################
                //! The std::array native pointer get trait specialization.
                template<
                    typename TElem,
                    std::size_t Tsize>
                struct GetPtrNative<
                    std::array<TElem, Tsize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        std::array<TElem, Tsize> const & view)
                    -> TElem const *
                    {
                        return view.data();
                    }
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPtrNative(
                        std::array<TElem, Tsize> & view)
                    -> TElem *
                    {
                        return view.data();
                    }
                };

                //#############################################################################
                //! The std::array pitch get trait specialization.
                template<
                    typename TElem,
                    std::size_t Tsize>
                struct GetPitchBytes<
                    dim::DimInt<0u>,
                    std::array<TElem, Tsize>>
                {
                    //-----------------------------------------------------------------------------
                    ALPAKA_FN_HOST static auto getPitchBytes(
                        std::array<TElem, Tsize> const & pitch)
                    -> idx::Idx<std::array<TElem, Tsize>>
                    {
                        return sizeof(TElem) * pitch.size();
                    }
                };
            }
        }
    }
    namespace offset
    {
        namespace traits
        {
            //#############################################################################
            //! The std::array offset get trait specialization.
            template<
                typename TIdx,
                typename TElem,
                std::size_t Tsize>
            struct GetOffset<
                TIdx,
                std::array<TElem, Tsize>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_FN_HOST static auto getOffset(
                    std::array<TElem, Tsize> const &)
                -> idx::Idx<std::array<TElem, Tsize>>
                {
                    return 0u;
                }
            };
        }
    }
    namespace idx
    {
        namespace traits
        {
            //#############################################################################
            //! The std::vector idx type trait specialization.
            template<
                typename TElem,
                std::size_t Tsize>
            struct IdxType<
                std::array<TElem, Tsize>>
            {
                using type = std::size_t;
            };
        }
    }
}
