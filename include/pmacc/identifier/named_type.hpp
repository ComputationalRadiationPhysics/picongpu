/* Copyright 2014-2021 Rene Widera
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

#include "pmacc/types.hpp"
#include "pmacc/identifier/identifier.hpp"
#include <string>

/* No namespace is needed because we only have defines*/

/** define a unique identifier with name, type and a default value
 * @param in_type type of the value
 * @param name name of identifier
 *
 * The created identifier has the following options:
 *          getName()         - return the name of the identifier
 *          ::type            - get contained type
 *
 * e.g. named_type(float,length)
 *      typedef length::type value_type; // is float
 *      printf("Identifier name: %s",length::getName()); //print Identifier name: length
 *
 * to create a instance of this value_identifier you can use:
 *      length();   or length_
 *
 */
#define named_type(in_type, name, ...)                                                                                \
    identifier(                                                                                                       \
        name, typedef in_type type; static std::string getName() { return std::string(#name); } __VA_ARGS__)
