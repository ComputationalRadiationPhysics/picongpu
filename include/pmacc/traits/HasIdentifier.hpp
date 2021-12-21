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
         * ::type (boost::mpl::bool_<>)
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
        bool hasIdentifier(const T_Object& obj, const T_Key& key)
        {
            return HasIdentifier<T_Object, T_Key>::type::value;
        }

    } // namespace traits

} // namespace pmacc
