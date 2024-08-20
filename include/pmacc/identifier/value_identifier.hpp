/* Copyright 2013-2023 Rene Widera, Pawel Ordyna
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/identifier/identifier.hpp"
#include "pmacc/types.hpp"

#include <string>

/* No namespace is needed because we only have defines*/

/** define a unique identifier with name, type and a default value
 * @param in_type type of the value
 * @param name name of identifier
 *
 * The created identifier has the following options:
 *      getValue()        - return the user defined value
 *      getName()         - return the name of the identifier
 *      ::type            - get type of the value
 *
 * e.g. value_identifier(float,length,0.0f)
 *      typedef length::type value_type; // is float
 *      value_type x = length::getValue(); //set x to 0.f
 *      printf("Identifier name: %s",length::getName()); //print Identifier name: length
 *
 * to create a instance of this value_identifier you can use:
 *      `length()` or `length_`
 * @{
 */

/** @param ... must be a device function/lambda
 *
 * @attention: getValue() is only callable from a pure device function e.g.
 *
 * @code{.cpp}
 * [] ALPAKA_FN_ACC() { return IdProvider<simDim>::getNewId();}
 * @endcode
 */
#define value_identifier_func(in_type, name, ...)                                                                     \
    identifier(                                                                                                       \
        name, using type = in_type; DINLINE static type getValue()                                                    \
        {                                                                                                             \
            auto const func = __VA_ARGS__;                                                                            \
            return func();                                                                                            \
        } static std::string getName() { return std::string(#name); })

/** getValue() is defined constexpr
 *  @param ... user defined value of in_type (can be a constructor of a class) e.g.
 *
 * @code{.cpp}
 * float3_X::create(0._X)
 * @endcode
 * @}
 */
#define value_identifier(in_type, name, ...)                                                                          \
    identifier(                                                                                                       \
        name, using type = in_type; HDINLINE static constexpr type getValue()                                         \
        { return __VA_ARGS__; } static std::string getName() { return std::string(#name); })
