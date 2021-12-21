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
#include <alpaka/dim/Traits.hpp>
#include <alpaka/vec/Vec.hpp>

#include <utility>

namespace alpaka
{
    namespace meta
    {
        namespace detail
        {
            //#############################################################################
            //! N-dimensional loop iteration template.
            template<typename TIndexSequence>
            struct NdLoop;
            //#############################################################################
            //! N-dimensional loop iteration template.
            template<>
            struct NdLoop<std::index_sequence<>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<typename TIndex, typename TExtentVec, typename TFnObj>
                ALPAKA_FN_HOST_ACC static auto ndLoop(TIndex& idx, TExtentVec const& extent, TFnObj const& f) -> void
                {
                    alpaka::ignore_unused(idx);
                    alpaka::ignore_unused(extent);
                    alpaka::ignore_unused(f);
                }
            };
            //#############################################################################
            //! N-dimensional loop iteration template.
            template<std::size_t Tdim>
            struct NdLoop<std::index_sequence<Tdim>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<typename TIndex, typename TExtentVec, typename TFnObj>
                ALPAKA_FN_HOST_ACC static auto ndLoop(TIndex& idx, TExtentVec const& extent, TFnObj const& f) -> void
                {
                    static_assert(
                        Dim<TIndex>::value > 0u,
                        "The dimension given to ndLoop has to be larger than zero!");
                    static_assert(
                        Dim<TIndex>::value == Dim<TExtentVec>::value,
                        "The dimensions of the iteration vector and the extent vector have to be identical!");
                    static_assert(
                        Dim<TIndex>::value > Tdim,
                        "The current dimension has to be in the range [0,dim-1]!");

                    for(idx[Tdim] = 0u; idx[Tdim] < extent[Tdim]; ++idx[Tdim])
                    {
                        f(idx);
                    }
                }
            };
            //#############################################################################
            //! N-dimensional loop iteration template.
            template<std::size_t Tdim0, std::size_t Tdim1, std::size_t... Tdims>
            struct NdLoop<std::index_sequence<Tdim0, Tdim1, Tdims...>>
            {
                //-----------------------------------------------------------------------------
                ALPAKA_NO_HOST_ACC_WARNING
                template<typename TIndex, typename TExtentVec, typename TFnObj>
                ALPAKA_FN_HOST_ACC static auto ndLoop(TIndex& idx, TExtentVec const& extent, TFnObj const& f) -> void
                {
                    static_assert(
                        Dim<TIndex>::value > 0u,
                        "The dimension given to ndLoop has to be larger than zero!");
                    static_assert(
                        Dim<TIndex>::value == Dim<TExtentVec>::value,
                        "The dimensions of the iteration vector and the extent vector have to be identical!");
                    static_assert(
                        Dim<TIndex>::value > Tdim0,
                        "The current dimension has to be in the range [0,dim-1]!");

                    for(idx[Tdim0] = 0u; idx[Tdim0] < extent[Tdim0]; ++idx[Tdim0])
                    {
                        detail::NdLoop<std::index_sequence<Tdim1, Tdims...>>::template ndLoop(idx, extent, f);
                    }
                }
            };
        } // namespace detail
        //-----------------------------------------------------------------------------
        //! Loops over an n-dimensional iteration index variable calling f(idx, args...) for each iteration.
        //! The loops are nested in the order given by the index_sequence with the first element being the outermost
        //! and the last index the innermost loop.
        //!
        //! \param indexSequence A sequence of indices being a permutation of the values [0, dim-1].
        //! \param extent N-dimensional loop extent.
        //! \param f The function called at each iteration.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TExtentVec, typename TFnObj, std::size_t... Tdims>
        ALPAKA_FN_HOST_ACC auto ndLoop(
            std::index_sequence<Tdims...> const& indexSequence,
            TExtentVec const& extent,
            TFnObj const& f) -> void
        {
            alpaka::ignore_unused(indexSequence);

            static_assert(
                Dim<TExtentVec>::value > 0u,
                "The dimension of the extent given to ndLoop has to be larger than zero!");
            static_assert(
                meta::IntegerSequenceValuesInRange<
                    std::index_sequence<Tdims...>,
                    std::size_t,
                    0,
                    Dim<TExtentVec>::value>::value,
                "The values in the index_sequence have to be in the range [0,dim-1]!");
            static_assert(
                meta::IntegerSequenceValuesUnique<std::index_sequence<Tdims...>>::value,
                "The values in the index_sequence have to be unique!");

            auto idx(Vec<Dim<TExtentVec>, Idx<TExtentVec>>::zeros());

            detail::NdLoop<std::index_sequence<Tdims...>>::template ndLoop(idx, extent, f);
        }
        //-----------------------------------------------------------------------------
        //! Loops over an n-dimensional iteration index variable calling f(idx, args...) for each iteration.
        //! The loops are nested from index zero outmost to index (dim-1) innermost.
        //!
        //! \param extent N-dimensional loop extent.
        //! \param f The function called at each iteration.
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TExtentVec, typename TFnObj>
        ALPAKA_FN_HOST_ACC auto ndLoopIncIdx(TExtentVec const& extent, TFnObj const& f) -> void
        {
            ndLoop(std::make_index_sequence<Dim<TExtentVec>::value>(), extent, f);
        }
    } // namespace meta
} // namespace alpaka
