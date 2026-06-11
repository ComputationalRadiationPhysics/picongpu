/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */


#pragma once

#include "pmacc/static_assert.hpp"

namespace pmacc
{
    namespace traits
    {
        /** Checks if a Objects has an identifier
         *
         * @tparam T_Object any object (class or typename)
         * @tparam T_Key a class which is used as identifier
         *
         * This struct must define
         * ::type (pmacc::mp_bool_<>)
         */
        template<typename T_Object, typename T_Key>
        struct HasIdentifier
        {
            /* The compiler is allowed to evaluate an expression that does not depend on a template parameter
             * even if the class is never instantiated. In that case static assert is always
             * evaluated (e.g. with clang), this results in an error if the condition is false.
             * http://www.boost.org/doc/libs/1_60_0/doc/html/boost_staticassert.html
             *
             * A workaround is to add a template dependency to the expression.
             * `sizeof(ANY_TYPE) != 0` is always true and defers the evaluation.
             */
            PMACC_CASSERT_MSG_TYPE(
                ___HasIdentifier_is_not_specialized_for_T_Object,
                T_Object,
                false && (sizeof(T_Object) != 0));
        };

        template<typename T_Object, typename T_Key>
        bool hasIdentifier(T_Object const& obj, T_Key const& key)
        {
            return HasIdentifier<T_Object, T_Key>::type::value;
        }

    } // namespace traits

} // namespace pmacc
