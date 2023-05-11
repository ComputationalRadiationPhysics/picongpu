/* Copyright 2013-2022 Rene Widera
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

#include "pmacc/types.hpp"

#include <boost/mpl/apply.hpp>

namespace pmacc
{
    /** convert a list to a map
     *
     * @tparam T_List an mp_list.
     * @tparam T_UnaryOperator unary operator to translate type from the sequence
     * to a mpl pair
     * @tparam T_Accessor An unary lambda operator which is used before the type
     * from the sequence is passed to T_UnaryOperator
     * @return ::type mpl map
     */
    template<typename T_List, typename T_MakePairUnaryOperator>
    struct SeqToMap
    {
        template<typename X>
        using Op = typename boost::mpl::apply<T_MakePairUnaryOperator, X>::type;

        using ListOfTuples = mp_transform<Op, T_List>;
        using type = mp_fold<ListOfTuples, mp_list<>, mp_map_insert>;
    };
} // namespace pmacc
