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

/** @file
 *
 * collection of `TypeMemberPair`s
 *
 * type id + member descriptions* are combined in a pair (called: TypeMemberPair)
 * (typeId,(value_type,name,initValue,...))
 *   - typeID and name are the only necessary values
 *     e.g. (0,(_,myName,_))
 */

#pragma once

#include "pmacc/preprocessor/facilities.hpp"
#include "pmacc/math/ConstVector.hpp"
#include "pmacc/math/Vector.hpp"

#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/punctuation.hpp>
#include <boost/preprocessor/punctuation/comma.hpp>
#include <boost/preprocessor/seq/to_tuple.hpp>
#include <boost/preprocessor/array/to_list.hpp>
#include <boost/preprocessor/control/if.hpp>
#include <boost/preprocessor/seq/for_each.hpp>
#include <boost/preprocessor/seq/to_array.hpp>
#include <boost/preprocessor/comparison/equal.hpp>
#include <boost/preprocessor/facilities/expand.hpp>


/** create static const member vector that needs no memory inside of the struct
 *
 *   @param type type of an element (types containing a comma are not allowed (e.g. `Vector<type,dim>`)
 *               use `typedef Vector<type,dim> NewType;` to avoid this behavior
 *   @param name member variable name
 *   @param ... enumeration of init values
 *
 *   @code{.cpp}
 *     PMACC_C_VECTOR(float2_64, center_SI, 1.134e-5, 1.134e-5);
 *     // is syntactically equivalent to
 *     static const float2_64 center_SI = float2_64(1.134e-5, 1.134e-5);
 *   @endcode
 */
#define PMACC_C_VECTOR(type, name, ...)                                                                               \
    (0,                                                                                                               \
     (typename pmacc::traits::GetValueType<type>::type,                                                               \
      name,                                                                                                           \
      pmacc::traits::GetNComponents<type>::value,                                                                     \
      __VA_ARGS__))


/** create static const member vector that needs no memory inside of the struct
 *
 *   @param type type of an element
 *   @param dim number of vector components
 *   @param name member variable name
 *   @param ... enumeration of init values (number of components must be greater or equal than dim)
 *
 *   @code{.cpp}
 *     PMACC_C_VECTOR_DIM(float_64, simDim, center_SI, 1.134e-5, 1.134e-5, 1.134e-5);
 *     // is syntactically equivalent to
 *     static const Vector<float_64,simDim> center_SI = Vector<float_64,simDim>(1.134e-5, 1.134e-5, 1.134e-5);
 *   @endcode
 */
#define PMACC_C_VECTOR_DIM(type, dim, name, ...) (0, (type, name, dim, __VA_ARGS__))

/** create static constexpr member
 *
 *   @param type type of the member
 *   @param name member variable name
 *   @param value init value
 *
 *   @code{.cpp}
 *     PMACC_C_VALUE(float_64, power_SI, 2.0);
 *     // is syntactically equivalent to
 *     static constexpr float_64 power_SI = float_64(2.0);
 *   @endcode
 */
#define PMACC_C_VALUE(type, name, value) (1, (type, name, value))

/** create changeable member
 *
 *   @param type type of the member
 *   @param name member variable name
 *   @param value init value
 *
 *   @code{.cpp}
 *     PMACC_VALUE(float_64, power_SI, 2.0);
 *     // is the equivalent of
 *     float_64 power_SI(2.0);
 *   @endcode
 */
#define PMACC_VALUE(type, name, initValue) (2, (type, name, initValue))


/** create changeable member vector
 *
 *   @param type type of an element
 *   @param name member variable name
 *   @param ... enumeration of init values
 *
 *   @code{.cpp}
 *     PMACC_VECTOR(float2_64, center_SI, 1.134e-5, 1.134e-5);
 *     // is the equivalent of
 *     float2_64 center_SI(1.134e-5, 1.134e-5);
 *   @endcode
 */
#define PMACC_VECTOR(type, name, ...) (5, (type, name, type(__VA_ARGS__)))

/** create changeable member vector
 *
 *   @param type type of an element
 *   @param dim number of vector components
 *   @param name member variable name
 *   @param ... enumeration of init values (number of components must be equal to dim)
 *
 *   @code{.cpp}
 *     PMACC_VECTOR_DIM(float_64, simDim, center_SI, 1.134e-5, 1.134e-5, 1.134e-5);
 *     // is the equivalent of
 *     Vector<float_64,3> center_SI(1.134e-5, 1.134e-5, 1.134e-5);
 *   @endcode
 */
#define PMACC_VECTOR_DIM(type, dim, name, ...)                                                                        \
    (5, ((pmacc::math::Vector<type, dim>), name, pmacc::math::Vector<type, dim>(__VA_ARGS__)))

/** create static const character string
 *
 *   @param name member variable name
 *   @param char_string character string
 *
 *   @code{.cpp}
 *     PMACC_C_STRING(filename, "fooFile.txt");
 *     // is syntactically equivalent to
 *     static const char* filename = (char*)"fooFile.txt";
 *   @endcode
 */
#define PMACC_C_STRING(name, initValue) (3, (_, name, initValue))

/** create any code extension
 *
 *   @param ... any code
 *
 *   @code{.cpp}
 *     PMACC_EXTENT(typedef float FooFloat;)
 *     // is the equivalent of
 *     typedef float FooFloat;
 *   @endcode
 */
#define PMACC_EXTENT(...) (4, (_, _, __VA_ARGS__))


/** select member description
 *
 * @param selectTypeID searched type id
 * @param op preprocessor function that is called with `def`
 * @param typeID type id of the current processed element
 * @param def element that is used by `op`
 * @return result of `(op def)` if `selectTypeID == typeID`
 *                   `( )`      else
 */
#define PMACC_PP_X_SELECT_TYPEID(selectTypeID, op, typeID, def)                                                       \
    BOOST_PP_IF(BOOST_PP_EQUAL(typeID, selectTypeID), (op def), ())

/** select member description of a TypeMemberPair for a specific type id
 *
 * @param typeID searched type id
 * @param op preprocessor function that is called with the second element of the selected pair
 * @param ... preprocessor TypeMemberPair
 * @return result of `op(secound(...))` if type is selected
 *                   `( )`              else
 */
#define PMACC_PP_SELECT_TYPEID(typeID, op, ...)                                                                       \
    PMACC_PP_X_SELECT_TYPEID(typeID, op, PMACC_PP_DEFER_FIRST() __VA_ARGS__, PMACC_PP_DEFER_SECOND() __VA_ARGS__)


/** run macro which calls accessor on the given element
 *
 * - the secound parameter (data) of a  BOOST_PP_SEQ_FOR_EACH macro is used as accessor
 * - parentheses around the result of the accessor were removed
 *
 * @param r no user argument (used by boost)
 * @param accessor preprocessor function which accepts an element of the sequence
 * @param elem the current evaluated element of the sequence
 *
 * @{
 */
#define PMACC_PP_SEQ_MACRO_WITH_ACCESSOR(r, accessor, elem) PMACC_PP_REMOVE_PAREN(accessor(elem))

#define PMACC_PP_X_CREATE_C_VECTOR_DEF(data, type, name, dim, ...)                                                    \
    PMACC_CONST_VECTOR_DEF(type, dim, name, __VA_ARGS__);
#define PMACC_PP_CREATE_C_VECTOR_DEF(elem) PMACC_PP_SELECT_TYPEID(0, PMACC_PP_X_CREATE_C_VECTOR_DEF, elem)

#define PMACC_PP_X_CREATE_C_VECTOR_VARIABLE(data, type, name, dim, ...) const BOOST_PP_CAT(name, _t) name;
#define PMACC_PP_CREATE_C_VECTOR_VARIABLE(elem) PMACC_PP_SELECT_TYPEID(0, PMACC_PP_X_CREATE_C_VECTOR_VARIABLE, elem)

#define PMACC_PP_X_CREATE_VALUE_VARIABLE(data, type, name, ...) type name;
#define PMACC_PP_CREATE_VALUE_VARIABLE(elem) PMACC_PP_SELECT_TYPEID(2, PMACC_PP_X_CREATE_VALUE_VARIABLE, elem)

#define PMACC_PP_X_CREATE_VALUE_VARIABLE_WITH_PAREN(data, type, name, ...) PMACC_PP_REMOVE_PAREN(type) name;
#define PMACC_PP_CREATE_VALUE_VARIABLE_WITH_PAREN(elem)                                                               \
    PMACC_PP_SELECT_TYPEID(5, PMACC_PP_X_CREATE_VALUE_VARIABLE_WITH_PAREN, elem)

#define PMACC_PP_X_CREATE_C_VALUE_VARIABLE(data, type, name, ...) static constexpr type name = __VA_ARGS__;
#define PMACC_PP_CREATE_C_VALUE_VARIABLE(elem) PMACC_PP_SELECT_TYPEID(1, PMACC_PP_X_CREATE_C_VALUE_VARIABLE, elem)


#define PMACC_PP_X1_INIT_VALUE_VARIABLE(data, type, name, ...) (name(__VA_ARGS__))
#define PMACC_PP_X_INIT_VALUE_VARIABLE(elem) PMACC_PP_SELECT_TYPEID(2, PMACC_PP_X1_INIT_VALUE_VARIABLE, elem)

#define PMACC_PP_X_INIT_VALUE_VARIABLE_WITH_PAREN(elem)                                                               \
    PMACC_PP_SELECT_TYPEID(5, PMACC_PP_X1_INIT_VALUE_VARIABLE, elem)

#define PMACC_PP_X_CREATE_C_STRING_VARIABLE(data, type, name, ...) static constexpr const char* name = __VA_ARGS__;
#define PMACC_PP_CREATE_C_STRING_VARIABLE(elem) PMACC_PP_SELECT_TYPEID(3, PMACC_PP_X_CREATE_C_STRING_VARIABLE, elem)

#define PMACC_PP_X_CREATE_EXTENT(data, type, name, ...) __VA_ARGS__
#define PMACC_PP_CREATE_EXTENT(elem) PMACC_PP_SELECT_TYPEID(4, PMACC_PP_X_CREATE_EXTENT, elem)

#define PMACC_PP_X1_ADD_DATA_TO_TYPEDESCRIPTION_MACRO(data, first, second)                                            \
    ((first, (data, PMACC_PP_REMOVE_PAREN(second))))
#define PMACC_PP_X_ADD_DATA_TO_TYPEDESCRIPTION_MACRO(data, value)                                                     \
    PMACC_PP_CALL(PMACC_PP_X1_ADD_DATA_TO_TYPEDESCRIPTION_MACRO, (data, value))

/** @} */

#define PMACC_PP_ADD_DATA_TO_TYPEDESCRIPTION_MACRO(r, data, elem)                                                     \
    PMACC_PP_X_ADD_DATA_TO_TYPEDESCRIPTION_MACRO(data, PMACC_PP_REMOVE_PAREN(elem))

/** create constructor initialization of non static variables
 *
 * add an emty struct to the end of the sequences to avoid problems with empty sequences
 *
 * @param ... preprocessor sequence with TypeMemberPair's to inherit from
 */
#define PMACC_PP_INIT_VALUE_VARIABLES(op, emptyStruct, ...)                                                           \
    PMACC_PP_DEFER_REMOVE_PAREN()                                                                                     \
    (BOOST_PP_EXPAND(                                                                                                 \
        BOOST_PP_SEQ_TO_TUPLE(BOOST_PP_SEQ_FOR_EACH(PMACC_PP_SEQ_MACRO_WITH_ACCESSOR, op, __VA_ARGS__ emptyStruct))))

/** generate the definition of a struct
 *
 * @param namespace_name name of a unique namespace to avoid naming conflicts
 * @param name name of the struct
 * @param ... preprocessor sequence with TypeMemberPair's
 */
#define PMACC_PP_STRUCT_DEF(namespace_name, name, ...)                                                                \
    namespace namespace_name                                                                                          \
    {                                                                                                                 \
        BOOST_PP_SEQ_FOR_EACH(PMACC_PP_SEQ_MACRO_WITH_ACCESSOR, PMACC_PP_CREATE_C_VECTOR_DEF, __VA_ARGS__)            \
        struct EmptyStruct                                                                                            \
        {                                                                                                             \
        };                                                                                                            \
        struct EmptyStruct2                                                                                           \
        {                                                                                                             \
        };                                                                                                            \
        struct name                                                                                                   \
            : private EmptyStruct                                                                                     \
            , private EmptyStruct2                                                                                    \
        {                                                                                                             \
            name()                                                                                                    \
                : PMACC_PP_INIT_VALUE_VARIABLES(                                                                      \
                    PMACC_PP_X_INIT_VALUE_VARIABLE,                                                                   \
                    ((2, (a, b, EmptyStruct))),                                                                       \
                    __VA_ARGS__)                                                                                      \
                , PMACC_PP_INIT_VALUE_VARIABLES(                                                                      \
                      PMACC_PP_X_INIT_VALUE_VARIABLE_WITH_PAREN,                                                      \
                      ((5, (a, b, EmptyStruct2))),                                                                    \
                      __VA_ARGS__)                                                                                    \
            {                                                                                                         \
            }                                                                                                         \
                                                                                                                      \
            BOOST_PP_SEQ_FOR_EACH(PMACC_PP_SEQ_MACRO_WITH_ACCESSOR, PMACC_PP_CREATE_C_VALUE_VARIABLE, __VA_ARGS__)    \
            BOOST_PP_SEQ_FOR_EACH(PMACC_PP_SEQ_MACRO_WITH_ACCESSOR, PMACC_PP_CREATE_VALUE_VARIABLE, __VA_ARGS__)      \
            BOOST_PP_SEQ_FOR_EACH(PMACC_PP_SEQ_MACRO_WITH_ACCESSOR, PMACC_PP_CREATE_C_VECTOR_VARIABLE, __VA_ARGS__)   \
            BOOST_PP_SEQ_FOR_EACH(PMACC_PP_SEQ_MACRO_WITH_ACCESSOR, PMACC_PP_CREATE_C_STRING_VARIABLE, __VA_ARGS__)   \
            BOOST_PP_SEQ_FOR_EACH(PMACC_PP_SEQ_MACRO_WITH_ACCESSOR, PMACC_PP_CREATE_EXTENT, __VA_ARGS__)              \
            BOOST_PP_SEQ_FOR_EACH(                                                                                    \
                PMACC_PP_SEQ_MACRO_WITH_ACCESSOR,                                                                     \
                PMACC_PP_CREATE_VALUE_VARIABLE_WITH_PAREN,                                                            \
                __VA_ARGS__)                                                                                          \
        };                                                                                                            \
    } /*namespace*/                                                                                                   \
    using namespace_name::name


/** add data to TypeMemberPair's
 *
 * transform (typeId,(value_type,name,initValue,...) to (typeId,(data,value_type,name,initValue,...)
 *
 * @param data any data which should be added to the TypeMemberPair's
 * @param ... preprocessor sequence with TypeMemberPair's
 */
#define PMACC_PP_ADD_DATA_TO_TYPEDESCRIPTION(data, ...)                                                               \
    BOOST_PP_SEQ_FOR_EACH(PMACC_PP_ADD_DATA_TO_TYPEDESCRIPTION_MACRO, data, __VA_ARGS__)

/** generate a struct with static and dynamic members
 *
 * @param name name of the struct
 * @param ... preprocessor sequence with TypeMemberPair's e.g. (PMACC_C_VALUE(int,a,2))
 *
 * @note do not forget the surrounding parenthesize for each element of a sequence
 *
 * @code{.cpp}
 * PMACC_STRUCT(StructAlice,
 *     // constant member variable
 *     (PMACC_C_VALUE(float, varFoo, -1.0))
 *     // lvalue member variable
 *     (PMACC_VALUE(float, varFoo, -1.0))
 *     // constant vector member variable
 *     (PMACC_C_VECTOR_DIM(double, 3, vectorBarC, 1.134e-5, 1.134e-5, 1.134e-5))
 *     // lvalue vector member variable
 *     (PMACC_VECTOR_DIM(double, 3, vectorBarC, 1.134e-5, 1.134e-5, 1.134e-5))
 *     // constant string member variable
 *     (PMACC_C_STRING(someString, "anythingYouWant: even spaces!"))
 *     // plain C++ member
 *     PMACC_EXTENT(
 *         using float_64 = double;
 *         static constexpr int varBar = 42;
 *     );
 * );
 * @endcode
 */
#define PMACC_STRUCT(name, ...)                                                                                       \
    PMACC_PP_STRUCT_DEF(                                                                                              \
        BOOST_PP_CAT(BOOST_PP_CAT(pmacc_, name), __COUNTER__),                                                        \
        name,                                                                                                         \
        PMACC_PP_ADD_DATA_TO_TYPEDESCRIPTION(name, __VA_ARGS__))
