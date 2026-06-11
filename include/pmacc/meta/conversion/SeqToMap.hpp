/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/types.hpp"

#include <boost/mpl/apply.hpp>

namespace pmacc
{
    /** convert a list to a map
     *
     * @tparam T_List an mp_list.
     * @tparam T_MakePairUnaryOperator unary operator to translate type from the sequence
     * to a mpl pair
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
