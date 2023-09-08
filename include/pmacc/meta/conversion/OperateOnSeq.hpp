/* Copyright 2015-2023 Rene Widera
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

#include "pmacc/meta/Mp11.hpp"
#include "pmacc/meta/accessors/Identity.hpp"
#include "pmacc/types.hpp"

#include <boost/mpl/apply.hpp>

namespace pmacc
{
    /** run an unary operator on each element of a sequence
     *
     * @tparam T_MPLSeq any boost mpl sequence
     * @tparam T_UnaryOperator unary operator to translate type from the sequence
     * to a mpl pair
     * @tparam T_Accessor an unary lambda operator that is used before the type
     * from the sequence is passed to T_UnaryOperator
     * @return ::type mp_list
     */
    template<typename T_MPLSeq, typename T_UnaryOperator, typename T_Accessor = meta::accessors::Identity<>>
    struct OperateOnSeq
    {
        template<typename X>
        using Op =
            typename boost::mpl::apply1<T_UnaryOperator, typename boost::mpl::apply1<T_Accessor, X>::type>::type;

        using type = mp_transform<Op, T_MPLSeq>;
    };

} // namespace pmacc
