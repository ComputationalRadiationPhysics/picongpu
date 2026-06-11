/*
 * SPDX-FileCopyrightText: Rene Widera
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "pmacc/static_assert.hpp"

#include <cstdint>
#include <type_traits>

namespace pmacc
{
    namespace traits
    {
        /** C
         *
         * @tparam T_Type any type
         * @return \p ::value as public with number of components (uint32_t)
         */
        template<typename T_Type, bool T_IsFundamental = std::is_fundamental_v<T_Type>>
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
