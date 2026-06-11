/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/static_assert.hpp"
#include "pmacc/types.hpp"

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
                using type = boost::mpl::void_;
            };
        };

    } // namespace errorHandlerPolicies
} // namespace pmacc
