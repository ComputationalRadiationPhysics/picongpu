/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/vec/Vec.hpp"

#include <utility>

namespace alpaka::meta
{
    namespace detail
    {
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TIndex, typename TExtentVec, typename TFnObj>
        ALPAKA_FN_HOST_ACC constexpr void ndLoopImpl(
            std::index_sequence<>,
            TIndex& idx,
            TExtentVec const&,
            TFnObj const& f)
        {
            f(idx);
        }

        ALPAKA_NO_HOST_ACC_WARNING
        template<std::size_t Tdim0, std::size_t... Tdims, typename TIndex, typename TExtentVec, typename TFnObj>
        ALPAKA_FN_HOST_ACC constexpr void ndLoopImpl(
            std::index_sequence<Tdim0, Tdims...>,
            TIndex& idx,
            TExtentVec const& extent,
            TFnObj const& f)
        {
            static_assert(Dim<TIndex>::value > 0u, "The dimension given to ndLoop has to be larger than zero!");
            static_assert(
                Dim<TIndex>::value == Dim<TExtentVec>::value,
                "The dimensions of the iteration vector and the extent vector have to be identical!");
            static_assert(Dim<TIndex>::value > Tdim0, "The current dimension has to be in the range [0,dim-1]!");

            for(idx[Tdim0] = 0u; idx[Tdim0] < extent[Tdim0]; ++idx[Tdim0])
            {
                ndLoopImpl(std::index_sequence<Tdims...>{}, idx, extent, f);
            }
        }
    } // namespace detail

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
        [[maybe_unused]] std::index_sequence<Tdims...> indexSequence,
        TExtentVec const& extent,
        TFnObj const& f) -> void
    {
        static_assert(
            IntegerSequenceValuesInRange<std::index_sequence<Tdims...>, std::size_t, 0, Dim<TExtentVec>::value>::value,
            "The values in the index_sequence have to be in the range [0,dim-1]!");
        static_assert(
            IntegerSequenceValuesUnique<std::index_sequence<Tdims...>>::value,
            "The values in the index_sequence have to be unique!");

        auto idx = Vec<Dim<TExtentVec>, Idx<TExtentVec>>::zeros();
        detail::ndLoopImpl(std::index_sequence<Tdims...>{}, idx, extent, f);
    }

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
} // namespace alpaka::meta
