/* Copyright 2022 Benjamin Worpitz, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/core/Common.hpp"
#include "alpaka/dim/Traits.hpp"
#include "alpaka/idx/Traits.hpp"
#include "alpaka/meta/IntegerSequence.hpp"

#include <utility>

namespace alpaka
{
    //! The vec traits.
    namespace trait
    {
        //! Trait for selecting a sub-vector.
        template<typename TVec, typename TIndexSequence, typename TSfinae = void>
        struct SubVecFromIndices;

        //! Trait for casting a vector.
        template<typename TVal, typename TVec, typename TSfinae = void>
        struct CastVec;

        //! Trait for reversing a vector.
        template<typename TVec, typename TSfinae = void>
        struct ReverseVec;

        //! Trait for concatenating two vectors.
        template<typename TVecL, typename TVecR, typename TSfinae = void>
        struct ConcatVec;
    } // namespace trait

    //! Builds a new vector by selecting the elements of the source vector in the given order.
    //! Repeating and swizzling elements is allowed.
    //! \return The sub-vector consisting of the elements specified by the indices.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TIndexSequence, typename TVec>
    ALPAKA_FN_HOST_ACC constexpr auto subVecFromIndices(TVec const& vec)
    {
        return trait::SubVecFromIndices<TVec, TIndexSequence>::subVecFromIndices(vec);
    }

    //! \tparam TVec has to specialize SubVecFromIndices.
    //! \return The sub-vector consisting of the first N elements of the source vector.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TSubDim, typename TVec>
    ALPAKA_FN_HOST_ACC constexpr auto subVecBegin(TVec const& vec)
    {
        static_assert(
            TSubDim::value <= Dim<TVec>::value,
            "The sub-Vec has to be smaller (or same size) then the original Vec.");

        //! A sequence of integers from 0 to dim-1.
        using IdxSubSequence = std::make_integer_sequence<std::size_t, TSubDim::value>;
        return subVecFromIndices<IdxSubSequence>(vec);
    }

    //! \tparam TVec has to specialize SubVecFromIndices.
    //! \return The sub-vector consisting of the last N elements of the source vector.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TSubDim, typename TVec>
    ALPAKA_FN_HOST_ACC constexpr auto subVecEnd(TVec const& vec)
    {
        static_assert(
            TSubDim::value <= Dim<TVec>::value,
            "The sub-Vec has to be smaller (or same size) then the original Vec.");

        constexpr std::size_t idxOffset = Dim<TVec>::value - TSubDim::value;

        //! A sequence of integers from 0 to dim-1.
        using IdxSubSequence = meta::MakeIntegerSequenceOffset<std::size_t, idxOffset, TSubDim::value>;
        return subVecFromIndices<IdxSubSequence>(vec);
    }

    //! \return The casted vector.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TVal, typename TVec>
    ALPAKA_FN_HOST_ACC constexpr auto castVec(TVec const& vec)
    {
        return trait::CastVec<TVal, TVec>::castVec(vec);
    }

    //! \return The reverseVec vector.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TVec>
    ALPAKA_FN_HOST_ACC constexpr auto reverseVec(TVec const& vec)
    {
        return trait::ReverseVec<TVec>::reverseVec(vec);
    }

    //! \return The concatenated vector.
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TVecL, typename TVecR>
    ALPAKA_FN_HOST_ACC constexpr auto concatVec(TVecL const& vecL, TVecR const& vecR)
    {
        return trait::ConcatVec<TVecL, TVecR>::concatVec(vecL, vecR);
    }
} // namespace alpaka
