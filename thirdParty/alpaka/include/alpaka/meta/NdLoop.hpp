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

#include <alpaka/vec/Vec.hpp>
#include <alpaka/dim/Traits.hpp>
#include <alpaka/meta/IntegerSequence.hpp>
#include <alpaka/core/Common.hpp>

#if !BOOST_ARCH_CUDA_DEVICE
    #include <boost/core/ignore_unused.hpp>
#endif

namespace alpaka
{
    namespace meta
    {
        namespace detail
        {
            //#############################################################################
            //! N-dimensional loop iteration template.
            template<
                typename TIndexSequence>
            struct NdLoop;
            //#############################################################################
            //! N-dimensional loop iteration template.
            template<>
            struct NdLoop<
                meta::IndexSequence<>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TIndex,
                    typename TExtentVec,
                    typename TFnObj>
                ALPAKA_FN_HOST_ACC static auto ndLoop(
#if !BOOST_ARCH_CUDA_DEVICE
                    TIndex & idx,
                    TExtentVec const & extent,
                    TFnObj const & f)
#else
                    TIndex &,
                    TExtentVec const &,
                    TFnObj const &)
#endif
                -> void
                {
#if !BOOST_ARCH_CUDA_DEVICE
                    boost::ignore_unused(idx);
                    boost::ignore_unused(extent);
                    boost::ignore_unused(f);
#endif
                }
            };
            //#############################################################################
            //! N-dimensional loop iteration template.
            template<
                std::size_t Tdim>
            struct NdLoop<
                meta::IndexSequence<Tdim>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TIndex,
                    typename TExtentVec,
                    typename TFnObj>
                ALPAKA_FN_HOST_ACC static auto ndLoop(
                    TIndex & idx,
                    TExtentVec const & extent,
                    TFnObj const & f)
                -> void
                {
                    static_assert(
                        dim::Dim<TIndex>::value > 0u,
                        "The dimension given to ndLoopIncIdx has to be larger than zero!");
                    static_assert(
                        dim::Dim<TIndex>::value == dim::Dim<TExtentVec>::value,
                        "The dimensions of the iteration vector and the extent vector have to be identical!");
                    static_assert(
                        dim::Dim<TIndex>::value > Tdim,
                        "The current dimension has to be in the rang [0,dim-1]!");

                    for(idx[Tdim] = 0u; idx[Tdim] < extent[Tdim]; ++idx[Tdim])
                    {
                        f(idx);
                    }
                }
            };
            //#############################################################################
            //! N-dimensional loop iteration template.
            template<
                std::size_t Tdim,
                std::size_t... Tdims>
            struct NdLoop<
                meta::IndexSequence<Tdim, Tdims...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<
                    typename TIndex,
                    typename TExtentVec,
                    typename TFnObj>
                ALPAKA_FN_HOST_ACC static auto ndLoop(
                    TIndex & idx,
                    TExtentVec const & extent,
                    TFnObj const & f)
                -> void
                {
                    static_assert(
                        dim::Dim<TIndex>::value > 0u,
                        "The dimension given to ndLoop has to be larger than zero!");
                    static_assert(
                        dim::Dim<TIndex>::value == dim::Dim<TExtentVec>::value,
                        "The dimensions of the iteration vector and the extent vector have to be identical!");
                    static_assert(
                        dim::Dim<TIndex>::value > Tdim,
                        "The current dimension has to be in the rang [0,dim-1]!");

                    for(idx[Tdim] = 0u; idx[Tdim] < extent[Tdim]; ++idx[Tdim])
                    {
                        detail::NdLoop<
                            meta::IndexSequence<Tdims...>>
                        ::template ndLoop(
                                idx,
                                extent,
                                f);
                    }
                }
            };
        }
        //-----------------------------------------------------------------------------
        //! Loops over an n-dimensional iteration index variable calling f(idx, args...) for each iteration.
        //! The loops are nested in the order given by the IndexSequence with the first element being the outermost and the last index the innermost loop.
        //!
#if !BOOST_ARCH_CUDA_DEVICE
        //! \param indexSequence A sequence of indices being a permutation of the values [0, dim-1], where every values occurs at most once.
#endif
        //! \param extent N-dimensional loop extent.
        //! \param f The function called at each iteration.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtentVec,
            typename TFnObj,
            std::size_t... Tdims>
        ALPAKA_FN_HOST_ACC auto ndLoop(
#if !BOOST_ARCH_CUDA_DEVICE
            meta::IndexSequence<Tdims...> const & indexSequence,
#else
            meta::IndexSequence<Tdims...> const &,
#endif
            TExtentVec const & extent,
            TFnObj const & f)
        -> void
        {
#if !BOOST_ARCH_CUDA_DEVICE
            boost::ignore_unused(indexSequence);
#endif

            static_assert(
                dim::Dim<TExtentVec>::value > 0u,
                "The dimension of the extent given to ndLoop has to be larger than zero!");
            static_assert(
                meta::IntegerSequenceValuesInRange<meta::IndexSequence<Tdims...>, std::size_t, 0, dim::Dim<TExtentVec>::value>::value,
                "The values in the IndexSequence have to be in the range [0,dim-1]!");
            static_assert(
                meta::IntegerSequenceValuesUnique<meta::IndexSequence<Tdims...>>::value,
                "The values in the IndexSequence have to be unique!");

            auto idx(
                vec::Vec<dim::Dim<TExtentVec>, size::Size<TExtentVec>>::zeros());

            detail::NdLoop<
                meta::IndexSequence<Tdims...>>
            ::template ndLoop(
                    idx,
                    extent,
                    f);
        }
        //-----------------------------------------------------------------------------
        //! Loops over an n-dimensional iteration index variable calling f(idx, args...) for each iteration.
        //! The loops are nested from index zero outmost to index (dim-1) innermost.
        //!
        //! \param extent N-dimensional loop extent.
        //! \param f The function called at each iteration.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TExtentVec,
            typename TFnObj>
        ALPAKA_FN_HOST_ACC auto ndLoopIncIdx(
            TExtentVec const & extent,
            TFnObj const & f)
        -> void
        {
            ndLoop(
                meta::MakeIndexSequence<dim::Dim<TExtentVec>::value>(),
                extent,
                f);
        }
    }
}
