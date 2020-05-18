/* Copyright 2019 Axel Huebl, Benjamin Worpitz
 *
 * This file is part of Alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#pragma once

#include <alpaka/core/Common.hpp>
#include <alpaka/core/Unused.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>
#include <alpaka/meta/Fold.hpp>
#include <alpaka/idx/Traits.hpp>

#include <type_traits>
#include <utility>
#include <functional>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The extent specifics.
    namespace extent
    {
        //-----------------------------------------------------------------------------
        //! The extent traits.
        namespace traits
        {
            //#############################################################################
            //! The extent get trait.
            //!
            //! If not specialized explicitly it returns 1.
            template<
                typename TIdxIntegralConst,
                typename TExtent,
                typename TSfinae = void>
            struct GetExtent
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtent const &)
                -> idx::Idx<TExtent>
                {
                    return static_cast<idx::Idx<TExtent>>(1);
                }
            };

            //#############################################################################
            //! The extent set trait.
            template<
                typename TIdxIntegralConst,
                typename TExtent,
                typename TExtentVal,
                typename TSfinae = void>
            struct SetExtent;
        }

        //-----------------------------------------------------------------------------
        //! \return The extent in the given dimension.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            std::size_t Tidx,
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto getExtent(
            TExtent const & extent = TExtent())
        -> idx::Idx<TExtent>
        {
            return
                traits::GetExtent<
                    dim::DimInt<Tidx>,
                    TExtent>
                ::getExtent(
                    extent);
        }
        //-----------------------------------------------------------------------------
        //! \return The width.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto getWidth(
            TExtent const & extent = TExtent())
        -> idx::Idx<TExtent>
        {
            return getExtent<dim::Dim<TExtent>::value - 1u>(extent);
        }
        //-----------------------------------------------------------------------------
        //! \return The height.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto getHeight(
            TExtent const & extent = TExtent())
        -> idx::Idx<TExtent>
        {
            return getExtent<dim::Dim<TExtent>::value - 2u>(extent);
        }
        //-----------------------------------------------------------------------------
        //! \return The depth.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto getDepth(
            TExtent const & extent = TExtent())
        -> idx::Idx<TExtent>
        {
            return getExtent<dim::Dim<TExtent>::value - 3u>(extent);
        }

        namespace detail
        {
            //-----------------------------------------------------------------------------
            ALPAKA_NO_HOST_ACC_WARNING
            template<
                typename TExtent,
                size_t... TIndices>
            ALPAKA_FN_HOST_ACC auto getExtentProductInternal(
                TExtent const & extent,
                std::index_sequence<TIndices...> const & indices)
            -> idx::Idx<TExtent>
            {
                alpaka::ignore_unused(indices);

                return
                    meta::foldr(
                        std::multiplies<idx::Idx<TExtent>>(),
                        getExtent<TIndices>(extent)...);
            }
        }

        //-----------------------------------------------------------------------------
        //! \return The product of the extent.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto getExtentProduct(
            TExtent const & extent = TExtent())
        -> idx::Idx<TExtent>
        {
            using IdxSequence = std::make_index_sequence<dim::Dim<TExtent>::value>;
            return
                detail::getExtentProductInternal(
                    extent,
                    IdxSequence());
        }

        //-----------------------------------------------------------------------------
        //! Sets the extent in the given dimension.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            std::size_t Tidx,
            typename TExtent,
            typename TExtentVal>
        ALPAKA_FN_HOST_ACC auto setExtent(
            TExtent & extent,
            TExtentVal const & extentVal)
        -> void
        {
            traits::SetExtent<
                dim::DimInt<Tidx>,
                TExtent,
                TExtentVal>
            ::setExtent(
                extent,
                extentVal);
        }
        //-----------------------------------------------------------------------------
        //! Sets the width.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtent,
            typename TWidth>
        ALPAKA_FN_HOST_ACC auto setWidth(
            TExtent & extent,
            TWidth const & width)
        -> void
        {
            setExtent<dim::Dim<TExtent>::value - 1u>(extent, width);
        }
        //-----------------------------------------------------------------------------
        //! Sets the height.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtent,
            typename THeight>
        ALPAKA_FN_HOST_ACC auto setHeight(
            TExtent & extent,
            THeight const & height)
        -> void
        {
            setExtent<dim::Dim<TExtent>::value - 2u>(extent, height);
        }
        //-----------------------------------------------------------------------------
        //! Sets the depth.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtent,
            typename TDepth>
        ALPAKA_FN_HOST_ACC auto setDepth(
            TExtent & extent,
            TDepth const & depth)
        -> void
        {
            setExtent<dim::Dim<TExtent>::value - 3u>(extent, depth);
        }

        //-----------------------------------------------------------------------------
        // Trait specializations for unsigned integral types.
        namespace traits
        {
            //#############################################################################
            //! The unsigned integral width get trait specialization.
            template<
                typename TExtent>
            struct GetExtent<
                dim::DimInt<0u>,
                TExtent,
                std::enable_if_t<
                    std::is_integral<TExtent>::value>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtent const & extent)
                -> idx::Idx<TExtent>
                {
                    return extent;
                }
            };
            //#############################################################################
            //! The unsigned integral width set trait specialization.
            template<
                typename TExtent,
                typename TExtentVal>
            struct SetExtent<
                dim::DimInt<0u>,
                TExtent,
                TExtentVal,
                std::enable_if_t<
                    std::is_integral<TExtent>::value>>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto setExtent(
                    TExtent const & extent,
                    TExtentVal const & extentVal)
                -> void
                {
                    extent = extentVal;
                }
            };
        }
    }
}
