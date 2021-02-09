/* Copyright 2015-2021 Rene Widera
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
#include "pmacc/static_assert.hpp"

namespace pmacc
{
    namespace errorHandlerPolicies
    {
        /** Throws an assertion that the value was not found in the sequence
         *  Binary meta function that takes any boost mpl sequence and a type
         */
        struct ThrowValueNotFound
        {
            template<typename T_MPLSeq, typename T_Value>
            struct apply
            {
                /* The compiler is allowed to evaluate an expression that does not depend on a template parameter
                 * even if the class is never instantiated. In that case static assert is always
                 * evaluated (e.g. with clang), this results in an error if the condition is false.
                 * http://www.boost.org/doc/libs/1_60_0/doc/html/boost_staticassert.html
                 *
                 * A workaround is to add a template dependency to the expression.
                 * `sizeof(ANY_TYPE) != 0` is always true and defers the evaluation.
                 */
                PMACC_CASSERT_MSG_TYPE(value_not_found_in_seq, T_Value, false && (sizeof(T_MPLSeq) != 0));
                typedef bmpl::void_ type;
            };
        };

    } // namespace errorHandlerPolicies
} // namespace pmacc
