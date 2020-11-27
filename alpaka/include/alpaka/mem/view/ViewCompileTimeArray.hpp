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
#include <alpaka/mem/buf/Traits.hpp>
#include <alpaka/pltf/PltfCpu.hpp>

#include <type_traits>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    // Trait specializations for fixed idx arrays.
    //
    // This allows the usage of multidimensional compile time arrays e.g. int[4][3] as argument to memory ops.
    /*namespace traits
    {
        //#############################################################################
        //! The fixed idx array device type trait specialization.
        template<
            typename TFixedSizeArray>
        struct DevType<
            TFixedSizeArray,
            std::enable_if_t<std::is_array<TFixedSizeArray>::value>>
        {
            using type = DevCpu;
        };

        //#############################################################################
        //! The fixed idx array device get trait specialization.
        template<
            typename TFixedSizeArray>
        struct GetDev<
            TFixedSizeArray,
            std::enable_if_t<std::is_array<TFixedSizeArray>::value>>
        {
            //-----------------------------------------------------------------------------
            ALPAKA_FN_HOST static auto getDev(
                TFixedSizeArray const & view)
            -> DevCpu
            {
                // \FIXME: CUDA device?
                return getDevByIdx<PltfCpu>(0u);
            }
        };

        //#############################################################################
        //! The fixed idx array dimension getter trait specialization.
        template<
            typename TFixedSizeArray>
        struct DimType<
            TFixedSizeArray,
            std::enable_if_t<std::is_array<TFixedSizeArray>::value>>
        {
            using type = DimInt<std::rank<TFixedSizeArray>::value>;
        };

        //#############################################################################
        //! The fixed idx array memory element type get trait specialization.
        template<
            typename TFixedSizeArray>
        struct ElemType<
            TFixedSizeArray,
            std::enable_if_t<
                std::is_array<TFixedSizeArray>::value>>
        {
            using type = std::remove_all_extent_t<TFixedSizeArray>;
        };
    }
    namespace extent
    {
        namespace traits
        {
            //#############################################################################
            //! The fixed idx array width get trait specialization.
            template<
                typename TIdxIntegralConst,
                typename TFixedSizeArray>
            struct GetExtent<
                TIdxIntegralConst,
                TFixedSizeArray,
                std::enable_if_t<
                    std::is_array<TFixedSizeArray>::value
                    && (std::rank<TFixedSizeArray>::value > TIdxIntegralConst::value)
                    && (std::extent<TFixedSizeArray, TIdxIntegralConst::value>::value > 0u)>>
            {
                //-----------------------------------------------------------------------------
                static constexpr auto getExtent(
                    TFixedSizeArray const & extent
                )
                -> Idx<TFixedSizeArray>
                {
                    alpaka::ignore_unused(extent);
                    return std::extent<TFixedSizeArray, TIdxIntegralConst::value>::value;
                }
            };
        }
    }
    namespace traits
    {
        //#############################################################################
        //! The fixed idx array native pointer get trait specialization.
        template<
            typename TFixedSizeArray>
        struct GetPtrNative<
            TFixedSizeArray,
            std::enable_if_t<
                std::is_array<TFixedSizeArray>::value>>
        {
            using TElem = std::remove_all_extent_t<TFixedSizeArray>;

            //-----------------------------------------------------------------------------
            static auto getPtrNative(
                TFixedSizeArray const & view)
            -> TElem const *
            {
                return view;
            }
            //-----------------------------------------------------------------------------
            static auto getPtrNative(
                TFixedSizeArray & view)
            -> TElem *
            {
                return view;
            }
        };

        //#############################################################################
        //! The fixed idx array pitch get trait specialization.
        template<
            typename TFixedSizeArray>
        struct GetPitchBytes<
            DimInt<std::rank<TFixedSizeArray>::value - 1u>,
            TFixedSizeArray,
            std::enable_if_t<
                std::is_array<TFixedSizeArray>::value
                && (std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value > 0u)>>
        {
            using TElem = std::remove_all_extent_t<TFixedSizeArray>;

            //-----------------------------------------------------------------------------
            static constexpr auto getPitchBytes(
                TFixedSizeArray const &)
            -> Idx<TFixedSizeArray>
            {
                return sizeof(TElem) * std::extent<TFixedSizeArray, std::rank<TFixedSizeArray>::value - 1u>::value;
            }
        };

        //#############################################################################
        //! The fixed idx array offset get trait specialization.
        template<
            typename TIdx,
            typename TFixedSizeArray>
        struct GetOffset<
            TIdx,
            TFixedSizeArray,
            std::enable_if_t<std::is_array<TFixedSizeArray>::value>>
        {
            //-----------------------------------------------------------------------------
            static auto getOffset(
                TFixedSizeArray const &)
            -> Idx<TFixedSizeArray>
            {
                return 0u;
            }
        };

        //#############################################################################
        //! The std::vector idx type trait specialization.
        template<
            typename TFixedSizeArray>
        struct IdxType<
            TFixedSizeArray,
            std::enable_if_t<std::is_array<TFixedSizeArray>::value>>
        {
            using type = std::size_t;
        };
    }*/
} // namespace alpaka
