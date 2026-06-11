/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
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
