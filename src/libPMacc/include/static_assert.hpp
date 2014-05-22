/**
 * Copyright 2013-2014 Axel Huebl, Felix Schmitt, Rene Widera
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */


#pragma once

#include "ppFunctions.hpp"
#include <boost/mpl/assert.hpp>

namespace PMacc
{
    /*type to create assert failures*/
    struct StaticAssertError
    {
    };
}

/** call BOOST_MPL_ASSERT_MSG and add unique id to message
 * @param cond an integral constant expression
 * @param msg a message those must a valid variable name (composition of characters_,A-Z,a-z)
 * @param unique_id pre compiler unique id
 */
#define PMACC_STATIC_ASSERT_MSG_DO2(cond,msg,unique_id)                        \
    BOOST_MPL_ASSERT_MSG(cond,PMACC_JOIN(msg,PMACC_JOIN(_________,unique_id)),(PMacc::StaticAssertError))


/*! static assert with error message
 * @param cond A condition which return true or false.
 * @param msg A message which is shown if the condition is false. Msg must a valid c++ variable name (etc. _only_human_make_mistakes)
 */
#define PMACC_STATIC_ASSERT_MSG(cond,msg) PMACC_STATIC_ASSERT_MSG_DO2(cond,msg,__COUNTER__)

/*! static assert
 * @param cond A condition which return true or false.
 */
#define PMACC_STATIC_ASSERT(cond) PMACC_STATIC_ASSERT_MSG_DO2(cond,STATIC_ASSERTION_FAILURE,__COUNTER__)

/*! static assert wrapper which is easier to use than \see PMACC_STATIC_ASSERT_MSG
 * @param msg A message which is shown if the condition is false. Msg must a valid c++ variable name (etc. _only_human_make_mistakes)
 * @param ... A condition which return true or false.
 */
#define PMACC_CASSERT_MSG(msg,...) PMACC_STATIC_ASSERT_MSG((__VA_ARGS__),msg)

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
#define PMACC_DEF_IN_NAMESPACE_MSG(msg,nmspace,var) \
  namespace msg {                       \
    using nmspace::var;                 \
    namespace fallback                  \
    {                                   \
      struct var                        \
      {                                 \
        double d[9999];                 \
        char   c;                       \
      };                                \
    }                                   \
    using fallback::var;                \
  }                                     \
  PMACC_CASSERT_MSG( msg, ((sizeof(msg::var))!=(sizeof(msg::fallback::var))) );
