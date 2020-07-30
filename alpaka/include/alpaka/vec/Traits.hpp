/* Copyright 2019 Benjamin Worpitz
 *
 * This file is part of alpaka.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */


#pragma once

#include <alpaka/dim/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/idx/Traits.hpp>
#include <alpaka/meta/IntegerSequence.hpp>

#include <utility>

namespace alpaka
{
    //-----------------------------------------------------------------------------
    //! The vec specifics.
    namespace vec
    {
        //-----------------------------------------------------------------------------
        //! The vec traits.
        namespace traits
        {
            //#############################################################################
            //! Trait for selecting a sub-vector.
            template<
                typename TVec,
                typename TIndexSequence,
                typename TSfinae = void>
            struct SubVecFromIndices;

            //#############################################################################
            //! Trait for casting a vector.
            template<
                typename TVal,
                typename TVec,
                typename TSfinae = void>
            struct Cast;

            //#############################################################################
            //! Trait for reversing a vector.
            template<
                typename TVec,
                typename TSfinae = void>
            struct Reverse;

            //#############################################################################
            //! Trait for concatenating two vectors.
            template<
                typename TVecL,
                typename TVecR,
                typename TSfinae = void>
            struct Concat;
        }

        //-----------------------------------------------------------------------------
        //! Builds a new vector by selecting the elements of the source vector in the given order.
        //! Repeating and swizzling elements is allowed.
        //! \return The sub-vector consisting of the elements specified by the indices.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TIndexSequence,
            typename TVec>
        ALPAKA_FN_HOST_ACC auto subVecFromIndices(
            TVec const & vec)
        {
            return
                traits::SubVecFromIndices<
                    TVec,
                    TIndexSequence>
                ::subVecFromIndices(
                    vec);
        }
        //-----------------------------------------------------------------------------
        //! \tparam TVec has to specialize SubVecFromIndices.
        //! \return The sub-vector consisting of the first N elements of the source vector.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TSubDim,
            typename TVec>
        ALPAKA_FN_HOST_ACC auto subVecBegin(
            TVec const & vec)
        {
            static_assert(
                TSubDim::value <= dim::Dim<TVec>::value,
                "The sub-Vec has to be smaller (or same size) then the original Vec.");

            //! A sequence of integers from 0 to dim-1.
            using IdxSubSequence =
                std::make_integer_sequence<
                    std::size_t,
                    TSubDim::value>;
            return
                subVecFromIndices<
                    IdxSubSequence>(
                        vec);
        }
        //-----------------------------------------------------------------------------
        //! \tparam TVec has to specialize SubVecFromIndices.
        //! \return The sub-vector consisting of the last N elements of the source vector.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TSubDim,
            typename TVec>
        ALPAKA_FN_HOST_ACC auto subVecEnd(
            TVec const & vec)
        {
            static_assert(
                TSubDim::value <= dim::Dim<TVec>::value,
                "The sub-Vec has to be smaller (or same size) then the original Vec.");

            constexpr std::size_t idxOffset = dim::Dim<TVec>::value - TSubDim::value;

            //! A sequence of integers from 0 to dim-1.
            using IdxSubSequence =
                meta::MakeIntegerSequenceOffset<
                    std::size_t,
                    idxOffset,
                    TSubDim::value>;
            return
                subVecFromIndices<
                    IdxSubSequence>(
                        vec);
        }

        //-----------------------------------------------------------------------------
        //! \return The casted vector.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TVal,
            typename TVec>
        ALPAKA_FN_HOST_ACC auto cast(
            TVec const & vec)
        {
            return
                traits::Cast<
                    TVal,
                    TVec>
                ::cast(
                    vec);
        }

        //-----------------------------------------------------------------------------
        //! \return The reverse vector.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TVec>
        ALPAKA_FN_HOST_ACC auto reverse(
            TVec const & vec)
        {
            return
                traits::Reverse<
                    TVec>
                ::reverse(
                    vec);
        }

        //-----------------------------------------------------------------------------
        //! \return The concatenated vector.
        ALPAKA_NO_HOST_ACC_WARNING
        template<
            typename TVecL,
            typename TVecR>
        ALPAKA_FN_HOST_ACC auto concat(
            TVecL const & vecL,
            TVecR const & vecR)
        {
            return
                traits::Concat<
                    TVecL,
                    TVecR>
                ::concat(
                    vecL,
                    vecR);
        }
    }
}
