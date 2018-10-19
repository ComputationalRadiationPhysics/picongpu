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

#include <alpaka/dim/Traits.hpp>
#include <alpaka/extent/Traits.hpp>
#include <alpaka/offset/Traits.hpp>
#include <alpaka/size/Traits.hpp>

#include <boost/config.hpp>

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
                typename TSize,
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
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::SubVecFromIndices<
                TVec,
                TIndexSequence>
            ::subVecFromIndices(
                vec))
#endif
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
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            subVecFromIndices<
                meta::MakeIntegerSequence<
                    std::size_t,
                    TSubDim::value
                >
            >(
                vec))
#endif
        {
            static_assert(
                TSubDim::value <= dim::Dim<TVec>::value,
                "The sub-Vec has to be smaller (or same size) then the original Vec.");

            //! A sequence of integers from 0 to dim-1.
            using IdxSubSequence =
                meta::MakeIntegerSequence<
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
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            subVecFromIndices<
                meta::MakeIntegerSequenceOffset<
                    std::size_t,
                    dim::Dim<TVec>::value - TSubDim::value,
                    TSubDim::value
                >
            >(
                vec))
#endif
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
            typename TSize,
            typename TVec>
        ALPAKA_FN_HOST_ACC auto cast(
            TVec const & vec)
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Cast<
                TSize,
                TVec>
            ::cast(
                vec))
#endif
        {
            return
                traits::Cast<
                    TSize,
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
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Reverse<
                TVec>
            ::reverse(
                vec))
#endif
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
#ifdef BOOST_NO_CXX14_RETURN_TYPE_DEDUCTION
        -> decltype(
            traits::Concat<
                TVecL,
                TVecR>
            ::concat(
                vecL,
                vecR))
#endif
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
