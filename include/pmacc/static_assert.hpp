/* Copyright 2013-2021 Axel Huebl, Felix Schmitt, Rene Widera
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

#include "pmacc/ppFunctions.hpp"
#include <boost/mpl/assert.hpp>

namespace pmacc
{
    /*type to create assert failures*/
    struct StaticAssertError
    {
    };

    template<typename T_Type = StaticAssertError>
    struct GetStaticAssertInfoType
    {
        typedef T_Type type;
    };
} // namespace pmacc

/** call BOOST_MPL_ASSERT_MSG and add unique id to message
 * @param pmacc_cond an integral constant expression
 * @param pmacc_msg a message which must a valid variable name (composition of characters_,A-Z,a-z)
 * @param pmacc_unique_id pre compiler unique id
 * @param pmacc_typeInfo a type that is shown in error message
 */
#if BOOST_LANG_CUDA && BOOST_COMP_CLANG_CUDA || BOOST_COMP_HIP
/* device compile with clang: boost static assert can not be used
 * error is: calling a `__host__` function from `__device__`
 * Therefore C++11 `static_assert` is used
 */
#    define PMACC_STATIC_ASSERT_MSG_DO2(pmacc_cond, pmacc_msg, pmacc_unique_id, pmacc_typeInfo)                       \
        static_assert(pmacc_cond, #pmacc_msg)
#else
#    define PMACC_STATIC_ASSERT_MSG_DO2(pmacc_cond, pmacc_msg, pmacc_unique_id, pmacc_typeInfo)                       \
        BOOST_MPL_ASSERT_MSG(                                                                                         \
            pmacc_cond,                                                                                               \
            PMACC_JOIN(pmacc_msg, PMACC_JOIN(_________, pmacc_unique_id)),                                            \
            (pmacc_typeInfo))
#endif

/*! static assert with error message
 * @param pmacc_cond A condition which return true or false.
 * @param pmacc_msg A message which is shown if the condition is false. Msg must a valid c++ variable name (etc.
 * _only_human_make_mistakes)
 * @param ... (optional) a type that is shown in error message
 */
#define PMACC_STATIC_ASSERT_MSG(pmacc_cond, pmacc_msg, ...)                                                           \
    PMACC_STATIC_ASSERT_MSG_DO2(                                                                                      \
        pmacc_cond,                                                                                                   \
        pmacc_msg,                                                                                                    \
        __COUNTER__,                                                                                                  \
        typename pmacc::GetStaticAssertInfoType<__VA_ARGS__>::type)

/*! static assert
 * @param pmacc_cond A condition which return true or false.
 */
#define PMACC_STATIC_ASSERT(pmacc_cond) PMACC_STATIC_ASSERT_MSG(pmacc_cond, STATIC_ASSERTION_FAILURE, )

/*! static assert wrapper which is easier to use than \see PMACC_STATIC_ASSERT_MSG
 * @param pmacc_msg A message which is shown if the condition is false. Msg must a valid c++ variable name (etc.
 * _only_human_make_mistakes)
 * @param pmacc_typeInfo a type that is shown in error message
 * @param ... A condition which return true or false.
 */
#define PMACC_CASSERT_MSG_TYPE(pmacc_msg, pmacc_typeInfo, ...)                                                        \
    PMACC_STATIC_ASSERT_MSG((__VA_ARGS__), pmacc_msg, pmacc_typeInfo)

/*! static assert wrapper which is easier to use than \see PMACC_STATIC_ASSERT_MSG
 * @param pmacc_msg A message which is shown if the condition is false. Msg must a valid c++ variable name (etc.
 * _only_human_make_mistakes)
 * @param ... A condition which return true or false.
 */
#define PMACC_CASSERT_MSG(pmacc_msg, ...) PMACC_STATIC_ASSERT_MSG((__VA_ARGS__), pmacc_msg, )

/*! static assert
 * @param ... A condition which return true or false.
 */
#define PMACC_CASSERT(...) PMACC_STATIC_ASSERT((__VA_ARGS__))

/*! static assert for undefined const variables
 *    using the SFINAE principle
 *
 * @param msg A message which is shown if the variable does not exist in the namespace
 * @param nmspace The name of the namespace
 * @param var The variable to look for.
 */
#define PMACC_DEF_IN_NAMESPACE_MSG(pmacc_msg, nmspace, var)                                                           \
    namespace pmacc_msg                                                                                               \
    {                                                                                                                 \
        using nmspace::var;                                                                                           \
        namespace fallback                                                                                            \
        {                                                                                                             \
            struct var                                                                                                \
            {                                                                                                         \
                double d[9999];                                                                                       \
                char c;                                                                                               \
            };                                                                                                        \
        }                                                                                                             \
        using fallback::var;                                                                                          \
    }                                                                                                                 \
    PMACC_CASSERT_MSG(pmacc_msg, ((sizeof(pmacc_msg::var)) != (sizeof(pmacc_msg::fallback::var))));
