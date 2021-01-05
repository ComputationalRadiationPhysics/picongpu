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

#include <boost/type_traits.hpp>

namespace pmacc
{
    namespace traits
    {
        /** C
         *
         * \tparam T_Type any type
         * \return \p ::value as public with number of components (uint32_t)
         */
        template<typename T_Type, bool T_IsFundamental = boost::is_fundamental<T_Type>::value>
        struct GetNComponents
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
                __GetNComponents_is_not_defined_for_this_type,
                T_Type,
                false && (sizeof(T_Type) != 0));
            static constexpr uint32_t value = 0;
        };

        /** return value=1 for al fundamental c++ types
         */
        template<typename T_Type>
        struct GetNComponents<T_Type, true>
        {
            static constexpr uint32_t value = 1;
        };

    } // namespace traits

} // namespace pmacc
