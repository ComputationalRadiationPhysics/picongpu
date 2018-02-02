/**
* \file
* Copyright 2014-2015 Benjamin Worpitz
*
* This file is part of alpaka.
*
* alpaka is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* alpaka is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with alpaka.
* If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <alpaka/meta/IntegerSequence.hpp>
#include <alpaka/core/Common.hpp>
#include <alpaka/meta/Fold.hpp>
#include <alpaka/size/Traits.hpp>
#include <alpaka/dim/DimIntegralConst.hpp>

#if !BOOST_ARCH_CUDA_DEVICE
    #include <boost/core/ignore_unused.hpp>
#endif

#include <type_traits>
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
                typename TIdx,
                typename TExtent,
                typename TSfinae = void>
            struct GetExtent
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtent const &)
                -> size::Size<TExtent>
                {
                    return static_cast<size::Size<TExtent>>(1);
                }
            };

            //#############################################################################
            //! The extent set trait.
            template<
                typename TIdx,
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
        -> size::Size<TExtent>
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
        -> size::Size<TExtent>
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
        -> size::Size<TExtent>
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
        -> size::Size<TExtent>
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
            ALPAKA_FN_HOST_ACC auto getProductOfExtentInternal(
                TExtent const & extent,
#if BOOST_ARCH_CUDA_DEVICE
                alpaka::meta::IndexSequence<TIndices...> const &)
#else
                alpaka::meta::IndexSequence<TIndices...> const & indices)
#endif
            -> size::Size<TExtent>
            {
#if !BOOST_ARCH_CUDA_DEVICE
                boost::ignore_unused(indices);
#endif
                return
                    meta::foldr(
                        std::multiplies<size::Size<TExtent>>(),
                        getExtent<TIndices>(extent)...);
            }
        }

        //-----------------------------------------------------------------------------
        //! \return The product of the extent.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtent>
        ALPAKA_FN_HOST_ACC auto getProductOfExtent(
            TExtent const & extent = TExtent())
        -> size::Size<TExtent>
        {
            using IdxSequence = alpaka::meta::MakeIndexSequence<dim::Dim<TExtent>::value>;
            return
                detail::getProductOfExtentInternal(
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
                typename std::enable_if<
                    std::is_integral<TExtent>::value>::type>
            {
                ALPAKA_NO_HOST_ACC_WARNING
                ALPAKA_FN_HOST_ACC static auto getExtent(
                    TExtent const & extent)
                -> size::Size<TExtent>
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
                typename std::enable_if<
                    std::is_integral<TExtent>::value>::type>
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
