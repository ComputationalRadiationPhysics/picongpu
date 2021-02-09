/* Copyright 2013-2021 Rene Widera
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include <boost/mpl/vector.hpp>
#include "pmacc/meta/conversion/ToSeq.hpp"
#include "pmacc/meta/conversion/JoinToSeq.hpp"
#include <boost/mpl/fold.hpp>

namespace pmacc
{
    /** combine all elements of the input type to a single vector
     *
     * If elements of the input sequence are a sequence themself, all of their
     * elements will be added to the resulting sequence
     *
     * @tparam T_In a boost mpl sequence or single type
     */
    template<typename T_In>
    struct MakeSeqFromNestedSeq
    {
    private:
        typedef typename ToSeq<T_In>::type Seq;

    public:
        typedef typename bmpl::fold<Seq, bmpl::vector0<>, JoinToSeq<bmpl::_1, bmpl::_2>>::type type;
    };

} // namespace pmacc
