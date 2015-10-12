/**
 * Copyright 2014-2015 Rene Widera, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "compileTime/conversion/SeqToMap.hpp"
#include "compileTime/conversion/TypeToAliasPair.hpp"
#include "compileTime/conversion/TypeToPair.hpp"
#include "compileTime/conversion/MakeSeqFromNestedSeq.hpp"
#include "compileTime/conversion/MakeSeq.hpp"
#include "math/Vector.hpp"
#include "types.h"

#include <boost/mpl/at.hpp>
#include <boost/mpl/copy.hpp>
#include <boost/mpl/pop_back.hpp>
#include <boost/type_traits/is_same.hpp>

namespace PMacc
{
namespace bmpl = boost::mpl;

/** Assign to each element in a sequence of CT::Vector(s) a type at a given
 *  component position
 *
 * @tparam T_InVector sequence with CT::Vector
 * @tparam T_ComponentPos position of the component to be changed (type must be bmpl::integral_c<uint32_t,X>)
 * @tparam T_Element value (type) which should replace the component at position T_Component
 *                   in the CT::Vector elements
 */
template<typename T_InVector,
typename T_ComponentPos,
typename T_Element
>
struct AssignToAnyElementInVector
{
    typedef T_InVector InVector;
    typedef T_Element Element;

    typedef typename bmpl::transform<
        InVector,
        PMacc::math::CT::Assign<bmpl::_1, T_ComponentPos, Element>
    >::type type;
};

namespace detail
{
/** Create tuples out of the elements of N sequences
 *
 * Combines all elements of N given sequences in T_MplSeq into N-tuples.
 * If the number of elements in each sequence is S0, S1, ... S(N-1)
 * than the resulting sequence will contain S0 * S1 * ... S(N-1) tuples.
 *
 * @tparam T_MplSeq sequence of input sequences
 * @tparam T_TmpResult temporary result
 * @tparam T_isEmpty true if T_MplSeq is empty else false
 */
template<typename T_MplSeq,
typename T_TmpResult = bmpl::vector0<>,
bool T_isEmpty = bmpl::empty<T_MplSeq>::value
>
struct AllCombinations;

/** implementation for inner recursive creation
 */
template<typename T_MplSeq, typename T_TmpResult>
struct AllCombinations<T_MplSeq, T_TmpResult, false >
{
    typedef T_MplSeq MplSeq;
    typedef T_TmpResult TmpResult;

    BOOST_STATIC_CONSTEXPR uint32_t rangeVectorSize = bmpl::size<MplSeq>::value;
    typedef typename bmpl::at<MplSeq, bmpl::integral_c<uint32_t, rangeVectorSize - 1 > > ::type LastElement;
    typedef typename MakeSeq<LastElement>::type LastElementAsSequence;
    typedef typename bmpl::pop_back<MplSeq>::type ShrinkedRangeVector;

    /* copy last given sequence to a mpl::vector to be sure that we can later on
     * call mpl::transform even if the input sequence is mpl::range_c
     */
    typedef typename bmpl::copy<LastElementAsSequence, bmpl::back_inserter< bmpl::vector0<> > >::type TmpVector;

    typedef typename bmpl::transform< TmpVector,
    AssignToAnyElementInVector<TmpResult,
    bmpl::integral_c<uint32_t, rangeVectorSize - 1 >, bmpl::_1 > >::type NestedSeq;


    typedef typename MakeSeqFromNestedSeq<NestedSeq>::type OneSeq;

    typedef typename detail::AllCombinations<ShrinkedRangeVector, OneSeq>::type type;
};

/** recursive end implementation
 */
template<typename T_MplSeq, typename T_TmpResult>
struct AllCombinations<T_MplSeq, T_TmpResult, true >
{
    typedef T_TmpResult type;
};

} //detail


/** Create tuples out of the elements of N sequences
 *
 * Combines all elements of N given sequences in T_MplSeq into N-tuples.
 * If the number of elements in each sequence is S0, S1, ... S(N-1)
 * than the resulting sequence will contain S0 * S1 * ... S(N-1) tuples.
 *
 * @tparam T_MplSeq N-dimensional sequence with input values
 *                  or single type (e.g. bmpl::integral_c<uint32_t,5>)
 *                  (if T_MplSeq is only one type it will be transformed to a sequence)
 * @tparam[out] ::type MplSequence of N-tuples
 *
 * example:
 *
 * sequence  == [ ]
 * tuple     == ( )
 *
 * T_MplSeq = [[1,2],[1],[4,3]]
 * combined to
 * ::type = [(1,1,4),(1,1,3),(2,1,4),(2,1,3)]
 */
template<typename T_MplSeq>
struct AllCombinations
{
    /* if T_MplSeq is no sequence it is a single type, we put this type in
     * a sequence because all next algorithms can only work with sequences */
    typedef typename MakeSeq<T_MplSeq>::type MplSeq;

    BOOST_STATIC_CONSTEXPR uint32_t rangeVectorSize = bmpl::size<MplSeq>::value;
    typedef typename bmpl::at<MplSeq, bmpl::integral_c<uint32_t, rangeVectorSize - 1 > > ::type LastElement;
    typedef typename MakeSeq<LastElement>::type LastElementAsSequence;

    typedef typename bmpl::pop_back<MplSeq>::type ShrinkedRangeVector;
    /* copy last given sequence to a mpl::vector to be sure that we can later on
     * call mpl::transform even if the input sequence is mpl::range_c
     */
    typedef typename bmpl::copy<LastElementAsSequence, bmpl::back_inserter< bmpl::vector0<> > >::type TmpVector;



    /* transform all elements in the vector to math::CT::vector<> */
    typedef math::CT::Vector<> EmptyVector;
    typedef typename bmpl::transform<
    TmpVector,
    PMacc::math::CT::Assign<EmptyVector, bmpl::integral_c<uint32_t, rangeVectorSize - 1 >, bmpl::_1>
    >::type FirstList;

    /* result type: MplSequence of N-tuples */
    typedef typename detail::AllCombinations<ShrinkedRangeVector, FirstList>::type type;
};


}//namespace PMacc
