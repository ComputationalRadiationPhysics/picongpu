/**
 * Copyright 2013 Axel Huebl, Felix Schmitt, René Widera
 *
 * This file is part of libPMacc. 
 * 
 * libPMacc is free software: you can redistribute it and/or modify 
 * it under the terms of of either the GNU General Public License or 
 * the GNU Lesser General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version. 
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
 
/* 
 * File:   static_assert.hpp
 * Author: widera, huebl
 *
 * Created on 13. August 2012, 13:30
 */

#ifndef STATIC_ASSERT_HPP
#define	STATIC_ASSERT_HPP

#include "ppFunctions.hpp"

namespace PMacc
{
    /*type to create assert failures*/
    struct StaticAssertError
    {
    };

    /*Switch between two types by a condition
     * @param Cond condition which must true or false 
     * @param T1 first type which set if condition is true
     * @param T2 second type which is set if condition is false
     * 
     * internel type name is "type"
     */
    template<bool Cond, typename T1, typename T2>
    struct GetTypeIf;

    template<typename T1, typename T2>
    struct GetTypeIf < true, T1, T2>
    {
        typedef T1 type;
    };

    template<typename T1, typename T2>
    struct GetTypeIf < false, T1, T2>
    {
        typedef T2 type;
    };
}

#define PMACC_STATIC_ASSERT_MSG_DO2(cond,msg,unique_id)               \
typedef ::PMacc::GetTypeIf<cond,const int,::PMacc::StaticAssertError>::type msg ;\
typedef const int TRUE_CONDITION;                                    \
TRUE_CONDITION PMACC_JOIN(TRUE_CONDITION,unique_id)=msg();       \
return PMACC_JOIN(TRUE_CONDITION,unique_id);

/*! Hide static assert in a struct to use it in function, global and namespace scope
 * nvcc has a special version For all other compiler there is a second 
 * BOOST_STATIC_ASSERT check to catch all asserts inside a struct/class which are
 * not tested by some compiler.
 */
#ifdef __CUDACC__

#define PMACC_STATIC_ASSERT_MSG_DO(cond,msg,unique_id) \
class PMACC_JOIN(PMACC_STATIC_ASSERT_,unique_id){ \
int msg (){ \
PMACC_STATIC_ASSERT_MSG_DO2(cond,msg,unique_id); \
} \
}
#else
#include <boost/static_assert.hpp>

#define PMACC_STATIC_ASSERT_MSG_DO(cond,msg,unique_id) \
class PMACC_JOIN(PMACC_STATIC_ASSERT_,unique_id){ \
int msg (){ \
PMACC_STATIC_ASSERT_MSG_DO2(cond,msg,unique_id); \
} \
BOOST_STATIC_ASSERT_MSG(cond,msg); \
}
#endif

/*! static assert with error message
 * @param cond A condition which return true or false.
 * @param msg A message which is shown if the condition is false. Msg must a valid c++ variable name (etc. _only_human_make_mistakes)
 */
#define PMACC_STATIC_ASSERT_MSG(cond,msg) PMACC_STATIC_ASSERT_MSG_DO(cond,msg,__COUNTER__)

/*! static assert
 * @param cond A condition which return true or false.
 */
#define PMACC_STATIC_ASSERT(cond) PMACC_STATIC_ASSERT_MSG(cond,STATIC_ASSERTION_FAILURE)

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
 * @param msg A message which is shown if the variable does not exist in the namespace nmspace
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

#endif	/* STATIC_ASSERT_HPP */

